// Worker-side model owner: lazy-load a model per quality key, run inference, return a square uint8
// mask (high = foreground). Owns ALL preprocessing so the worker stays model-agnostic.
//
// Two models, descriptor-driven (see models.ts + PHASE-CONTRACT 2b):
//   fast (U-2-Netp, 320²)      — manual ImageNet preprocess; transformers.js 4.2.0 CANNOT build its
//                                AutoProcessor (`Unknown image_processor_type: 'U2NetImageProcessor'`),
//                                so we squash to 320² and normalize by hand. Output `1959`, saliency 0..1.
//   hq   (BiRefNet_lite, 1024²) — AutoProcessor works (declares the built-in ViTFeatureExtractor); fed a
//                                RawImage built from the worker's already-EXIF-oriented ImageData (NOT
//                                RawImage.fromBlob) so HQ honors EXIF identically to fast + the full-res
//                                original. Output `output_image`, a single LOGIT plane (sigmoid in extract).
// Two resident slots are allowed (Map); execution stays serialized by main.ts's inFlight guard
// (single concurrent ORT execution — ARCHITECTURE §6). HQ is WebGPU-only (no WASM fallback, D4).
import { AutoModel, AutoProcessor, RawImage, Tensor, env } from "@huggingface/transformers";
import { MODELS, maskFromOutput, type QualityKey, type ModelDescriptor } from "./models.ts";

// ImageNet normalization (fast manual path only; hq's AutoProcessor normalizes internally).
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export type Backend = "webgpu" | "wasm";
export type ProgressCb = (p: unknown) => void;

type ModelFn = (inputs: Record<string, Tensor>) => Promise<Record<string, OutTensor>>;
interface OutTensor { data: Float32Array | Uint16Array | number[]; dims: number[]; }
type ProcessorFn = (img: RawImage) => Promise<{ pixel_values: Tensor }>;

interface Loaded {
  descriptor: ModelDescriptor;
  model: ModelFn;
  processor?: ProcessorFn; // hq only
  inputName: string;
  backend: Backend;
}

export interface SegmentResult {
  data: Uint8Array; // single-channel mask, length size*size
  size: number;     // mask edge in px (= descriptor.inputSize: 320 fast / 1024 hq)
}

// One load slot per quality key — both models may be resident at once.
const slots = new Map<QualityKey, Promise<Loaded>>();

/**
 * Configure transformers.js for a zero-backend static deploy. Must be called once before load.
 * @param ortBase absolute URL of the directory holding our self-hosted ORT wasm (e.g. ".../ort/")
 */
export function configureEnv(ortBase: string): void {
  const wasm = env.backends.onnx.wasm;
  if (wasm) {
    // Self-host the ORT wasm/glue (shipped in dist/ort/) instead of the default CDN.
    wasm.wasmPaths = ortBase;
    // GitHub Pages can't set COOP/COEP, so SharedArrayBuffer is unavailable -> single-threaded wasm.
    wasm.numThreads = 1;
  }
  // Model weights come from the HF hub at runtime; we never bundle/serve them ourselves.
  env.allowLocalModels = false;
}

async function detectWebGPU(): Promise<boolean> {
  try {
    const gpu = (globalThis.navigator as { gpu?: { requestAdapter(): Promise<unknown> } } | undefined)?.gpu;
    if (!gpu) return false;
    return (await gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

async function load(key: QualityKey, onProgress?: ProgressCb): Promise<Loaded> {
  let slot = slots.get(key);
  if (!slot) {
    slot = (async () => {
      const d = MODELS[key];
      const mk = (device: Backend) =>
        AutoModel.from_pretrained(d.id, { dtype: d.dtype, device, progress_callback: onProgress } as never);

      // Pick the backend. hq is WebGPU-only (D4) — fail loudly, never silently fall to a 1024²-WASM OOM.
      // fast is "auto": WebGPU only if an adapter actually exists, else WASM (with load-time fallback).
      let backend: Backend;
      let model: unknown;
      if (d.device === "webgpu") {
        backend = "webgpu";
        model = await mk("webgpu"); // no fallback for hq
      } else {
        backend = (await detectWebGPU()) ? "webgpu" : "wasm";
        try {
          model = await mk(backend);
        } catch (e1) {
          console.warn(`[ClearCut] ${backend} backend failed for ${key}, falling back to wasm:`, e1);
          backend = "wasm";
          model = await mk("wasm");
        }
      }

      const processor = d.preprocess === "auto-processor"
        ? ((await AutoProcessor.from_pretrained(d.id)) as unknown as ProcessorFn)
        : undefined;

      // Resolve the real ONNX input name rather than hardcoding (fast: "input.1", hq: "input_image").
      const sessions = (model as { sessions?: Record<string, { inputNames?: string[] }> }).sessions ?? {};
      const session = sessions.model ?? Object.values(sessions)[0];
      const fallbackName = d.preprocess === "auto-processor" ? "input_image" : "input.1";
      const inputName = session?.inputNames?.[0] ?? fallbackName;
      return { descriptor: d, model: model as ModelFn, processor, inputName, backend };
    })().catch((e) => {
      // Don't cache a rejected load — a transient download failure must not brick future attempts.
      slots.delete(key);
      throw e;
    });
    slots.set(key, slot);
  }
  return slot;
}

/** Load a model (lazy, cached per key). Returns the active backend for status reporting. */
export async function ensureModel(key: QualityKey, onProgress?: ProgressCb): Promise<Backend> {
  return (await load(key, onProgress)).backend;
}

// Read an output tensor's plane as fp32 numbers regardless of its on-wire dtype. An fp16 model can
// hand back raw fp16 bits (Uint16Array); reading those as JS numbers would corrupt the sigmoid, so
// convert through the Tensor's own .to("float32") in that case (A2 watch-item, PHASE-CONTRACT §10).
function asFloat32(tensor: OutTensor): Float32Array | number[] {
  const data = tensor.data;
  if (data instanceof Float32Array || Array.isArray(data)) return data;
  const converted = (tensor as unknown as { to(t: string): OutTensor }).to("float32");
  return converted.data as Float32Array;
}

/**
 * Segment an EXIF-oriented bitmap with the given model. Owns all preprocessing; closes the bitmap.
 * Returns a square uint8 mask (size = the model's input edge). composite.ts upscales it to full res.
 */
export async function segment(key: QualityKey, bitmap: ImageBitmap): Promise<SegmentResult> {
  const { model, processor, inputName, descriptor } = await load(key);
  try {
    return descriptor.preprocess === "manual-squash"
      ? segmentManual(model, inputName, descriptor, bitmap)
      : await segmentAuto(model, processor!, inputName, descriptor, bitmap);
  } finally {
    bitmap.close();
  }
}

// fast: squash-draw to 320², manual ÷255 + ImageNet normalize, NCHW [1,3,S,S]. Byte-identical to Phase 1.
function segmentManual(model: ModelFn, inputName: string, d: ModelDescriptor, bitmap: ImageBitmap): Promise<SegmentResult> {
  const S = d.inputSize;
  const canvas = new OffscreenCanvas(S, S);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("OffscreenCanvas 2D context unavailable in worker");
  ctx.drawImage(bitmap, 0, 0, S, S);
  const rgba = ctx.getImageData(0, 0, S, S).data;

  const plane = S * S;
  const chw = new Float32Array(3 * plane);
  for (let p = 0; p < plane; p++) {
    const r = rgba[p * 4] / 255;
    const g = rgba[p * 4 + 1] / 255;
    const b = rgba[p * 4 + 2] / 255;
    chw[p] = (r - MEAN[0]) / STD[0];
    chw[plane + p] = (g - MEAN[1]) / STD[1];
    chw[2 * plane + p] = (b - MEAN[2]) / STD[2];
  }
  const input = new Tensor("float32", chw, [1, 3, S, S]);
  return model({ [inputName]: input }).then((output) => {
    const sal = output[d.outputKey!] ?? output[Object.keys(output)[0]];
    return { data: maskFromOutput(d.output, asFloat32(sal)), size: S };
  });
}

// hq: native-size ImageData -> RawImage(.rgb) -> AutoProcessor (resizes to 1024² + ImageNet-normalizes
// internally) -> model -> single logit plane -> logit-sigmoid extraction. NO in-segment resize: the
// 1024² mask goes to composite.ts, which upscales to source dims (full-res invariant lives there).
async function segmentAuto(
  model: ModelFn, processor: ProcessorFn, inputName: string, d: ModelDescriptor, bitmap: ImageBitmap,
): Promise<SegmentResult> {
  const w = bitmap.width;
  const h = bitmap.height;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("OffscreenCanvas 2D context unavailable in worker");
  ctx.drawImage(bitmap, 0, 0); // native size, no scaling — the processor does the 1024² resize
  const img = ctx.getImageData(0, 0, w, h);

  const raw = new RawImage(img.data, w, h, 4).rgb(); // drop alpha -> 3 channels for the processor
  const { pixel_values } = await processor(raw);
  const output = await model({ [inputName]: pixel_values });

  const out = output["output_image"] ?? output[Object.keys(output)[0]];
  const dims = out.dims;
  const size = dims[dims.length - 1]; // square output edge (1024)
  return { data: maskFromOutput(d.output, asFloat32(out)), size };
}
