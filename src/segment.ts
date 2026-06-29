// Worker-side model owner: lazy-load a model per quality key, run inference, return a square uint8
// mask (high = foreground). Owns ALL preprocessing so the worker stays model-agnostic.
//
// Two models, two runtimes (see models.ts + PHASE-CONTRACT 2b Amendment 1):
//   fast (U-2-Netp, 320²)     — runtime "transformers": @huggingface/transformers AutoModel. Its
//                               AutoProcessor is broken on 4.2.0, so we preprocess by hand. WebGPU
//                               when an adapter exists, else WASM. Output "1959" (already 0..1).
//   hq   (IS-Net, 1024²)      — runtime "ort-raw": a RAW onnxruntime-web session over a bare Apache
//                               .onnx (transformers.js can't load it — no HF config). WASM only (the
//                               export's MaxPool ceil_mode is unsupported on ORT-Web WebGPU). The
//                               176 MB model is fetched with streaming progress. Output "output" 0..1.
// Preprocessing is unified + descriptor-driven (preprocessNCHW). Two resident slots are allowed (Map);
// execution stays serialized by main.ts's inFlight guard (single concurrent execution, ARCHITECTURE §6).
import { AutoModel, Tensor, env } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";
import { MODELS, maskFromOutput, preprocessNCHW, type QualityKey, type ModelDescriptor } from "./models.ts";

export type Backend = "webgpu" | "wasm";
export type ProgressCb = (p: unknown) => void;

interface OutTensor { data: Float32Array | Uint16Array | number[]; dims: number[]; }

interface Loaded {
  descriptor: ModelDescriptor;
  // Run normalized NCHW input through the model; return the primary output plane.
  run: (chw: Float32Array) => Promise<OutTensor>;
  backend: Backend;
}

export interface SegmentResult {
  data: Uint8Array; // single-channel mask, length size*size
  size: number;     // mask edge in px (= descriptor.inputSize: 320 fast / 1024 hq)
}

const slots = new Map<QualityKey, Promise<Loaded>>();

/**
 * Configure transformers.js AND raw onnxruntime-web for a zero-backend static deploy.
 * @param ortBase absolute URL of the directory holding our self-hosted ORT wasm (e.g. ".../ort/")
 */
export function configureEnv(ortBase: string): void {
  const wasm = env.backends.onnx.wasm;
  if (wasm) {
    wasm.wasmPaths = ortBase; // self-host the ORT wasm/glue (dist/ort/) instead of the CDN
    wasm.numThreads = 1;      // GitHub Pages can't set COOP/COEP -> no SharedArrayBuffer -> single-thread
  }
  // The raw ort-raw session uses ort.env directly; point it at the same self-hosted wasm.
  ort.env.wasm.wasmPaths = ortBase;
  ort.env.wasm.numThreads = 1;
  env.allowLocalModels = false; // transformers weights come from the HF hub at runtime
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

// Read an output tensor's plane as fp32 numbers regardless of its on-wire dtype (defensive).
function asFloat32(tensor: OutTensor): Float32Array | number[] {
  const data = tensor.data;
  if (data instanceof Float32Array || Array.isArray(data)) return data;
  const converted = (tensor as unknown as { to(t: string): OutTensor }).to("float32");
  return converted.data as Float32Array;
}

// Stream a model file with progress so the 176 MB IS-Net download shows a percentage (raw
// InferenceSession.create(url) gives no progress events). Emits the transformers.js progress shape.
async function fetchWithProgress(url: string, onProgress?: ProgressCb): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok || !resp.body) throw new Error(`model fetch failed: HTTP ${resp.status}`);
  const total = Number(resp.headers.get("content-length")) || 0;
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total && onProgress) onProgress({ status: "progress", progress: (received / total) * 100 });
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

// --- transformers runtime (fast / U-2-Netp) -------------------------------
async function loadTransformers(d: ModelDescriptor, onProgress?: ProgressCb): Promise<Loaded> {
  const mk = (device: Backend) =>
    AutoModel.from_pretrained(d.id!, { dtype: d.dtype, device, progress_callback: onProgress } as never);
  let backend: Backend = (await detectWebGPU()) ? "webgpu" : "wasm";
  let model: unknown;
  try {
    model = await mk(backend);
  } catch (e1) {
    console.warn(`[ClearCut] ${backend} backend failed for ${d.key}, falling back to wasm:`, e1);
    backend = "wasm";
    model = await mk("wasm");
  }
  const sessions = (model as { sessions?: Record<string, { inputNames?: string[] }> }).sessions ?? {};
  const session = sessions.model ?? Object.values(sessions)[0];
  const inputName = session?.inputNames?.[0] ?? "input.1";
  const m = model as (inputs: Record<string, Tensor>) => Promise<Record<string, OutTensor>>;
  const S = d.inputSize;
  const run = async (chw: Float32Array): Promise<OutTensor> => {
    const out = await m({ [inputName]: new Tensor("float32", chw, [1, 3, S, S]) });
    return out[d.outputKey!] ?? out[Object.keys(out)[0]];
  };
  return { descriptor: d, run, backend };
}

// --- ort-raw runtime (hq / IS-Net) ----------------------------------------
async function loadOrtRaw(d: ModelDescriptor, onProgress?: ProgressCb): Promise<Loaded> {
  const bytes = await fetchWithProgress(d.url!, onProgress);
  const session = await ort.InferenceSession.create(bytes, { executionProviders: [d.ep!] });
  const inputName = session.inputNames[0] ?? d.inputName!;
  const outputName = d.outputName && session.outputNames.includes(d.outputName)
    ? d.outputName
    : session.outputNames[0];
  const S = d.inputSize;
  const run = async (chw: Float32Array): Promise<OutTensor> => {
    const out = await session.run({ [inputName]: new ort.Tensor("float32", chw, [1, 3, S, S]) });
    const t = out[outputName] as unknown as OutTensor;
    return t;
  };
  return { descriptor: d, run, backend: d.ep as Backend };
}

async function load(key: QualityKey, onProgress?: ProgressCb): Promise<Loaded> {
  let slot = slots.get(key);
  if (!slot) {
    slot = (async () => {
      const d = MODELS[key];
      return d.runtime === "ort-raw" ? loadOrtRaw(d, onProgress) : loadTransformers(d, onProgress);
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

/**
 * Segment an EXIF-oriented bitmap with the given model. Owns all preprocessing; closes the bitmap.
 * Returns a square uint8 mask (size = the model's input edge). composite.ts upscales it to full res.
 */
export async function segment(key: QualityKey, bitmap: ImageBitmap): Promise<SegmentResult> {
  const { run, descriptor } = await load(key);
  try {
    const S = descriptor.inputSize;
    const canvas = new OffscreenCanvas(S, S);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("OffscreenCanvas 2D context unavailable in worker");
    ctx.drawImage(bitmap, 0, 0, S, S); // squash to the model's input size
    const rgba = ctx.getImageData(0, 0, S, S).data;
    const chw = preprocessNCHW(rgba, S, descriptor.rescale, descriptor.mean, descriptor.std);
    const out = await run(chw);
    return { data: maskFromOutput(descriptor.output, asFloat32(out)), size: S };
  } finally {
    bitmap.close();
  }
}
