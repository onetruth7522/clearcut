// Worker-side model owner: lazy-load U-2-Netp, run inference, return a 320x320 uint8 mask.
//
// NOTE (verified live 2026-06-25, see MEASUREMENTS.md): transformers.js 4.2.0 cannot build this
// model's AutoProcessor (`Unknown image_processor_type: 'U2NetImageProcessor'`), so preprocessing
// is done by hand here. The model has input `input.1` and emits fp32 outputs `1959`..`1965`
// (each [1,1,320,320], range 0..1); `1959` is the composite saliency map (high = foreground).
import { AutoModel, Tensor, env } from "@huggingface/transformers";

const MODEL_ID = "BritishWerewolf/U-2-Netp";
const S = 320;
// ImageNet normalization from the model's preprocessor_config.json.
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export type Backend = "webgpu" | "wasm";
export type ProgressCb = (p: unknown) => void;

interface Loaded {
  // transformers.js model instance; typed loosely — the API surface we use is small.
  model: (inputs: Record<string, Tensor>) => Promise<Record<string, { data: Float32Array; dims: number[] }>>;
  inputName: string;
  backend: Backend;
}

let loaded: Promise<Loaded> | null = null;

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

async function load(onProgress?: ProgressCb): Promise<Loaded> {
  if (!loaded) {
    loaded = (async () => {
      const mk = (device: Backend) =>
        AutoModel.from_pretrained(MODEL_ID, { dtype: "fp32", device, progress_callback: onProgress } as never);
      let model: unknown;
      let backend: Backend = "wasm";
      // Pick the backend up front: only ask for WebGPU if an adapter actually exists, so we
      // never trip ORT's "no available backend" when WebGPU is absent (headless, older browsers).
      const hasWebGPU = await detectWebGPU();
      try {
        backend = hasWebGPU ? "webgpu" : "wasm";
        model = await mk(backend);
      } catch (e1) {
        // Primary backend failed at load — fall back to wasm (e.g. WebGPU adapter lost).
        console.warn(`[ClearCut] ${backend} backend failed, falling back to wasm:`, e1);
        backend = "wasm";
        model = await mk("wasm");
      }
      // Resolve the real ONNX input name rather than hardcoding (it is "input.1" for this export).
      const sessions = (model as { sessions?: Record<string, { inputNames?: string[] }> }).sessions ?? {};
      const session = sessions.model ?? Object.values(sessions)[0];
      const inputName = session?.inputNames?.[0] ?? "input.1";
      return { model: model as Loaded["model"], inputName, backend };
    })().catch((e) => {
      // Don't cache a rejected load — a transient download failure must not brick every
      // future attempt until a full page reload. Reset so the next drop retries cleanly.
      loaded = null;
      throw e;
    });
  }
  return loaded;
}

/** Load the model (lazy, cached). Returns the active backend for status reporting. */
export async function ensureModel(onProgress?: ProgressCb): Promise<Backend> {
  return (await load(onProgress)).backend;
}

/**
 * Run segmentation on a 320x320 RGBA buffer (already squashed to the model's input size by the
 * caller) and return a 320x320 single-channel uint8 mask (length 102400). High = foreground.
 */
export async function segment(rgba320: Uint8ClampedArray): Promise<Uint8Array> {
  const { model, inputName } = await load();
  const plane = S * S;

  // Build normalized NCHW float32 tensor [1,3,320,320] from RGBA.
  const chw = new Float32Array(3 * plane);
  for (let p = 0; p < plane; p++) {
    const r = rgba320[p * 4] / 255;
    const g = rgba320[p * 4 + 1] / 255;
    const b = rgba320[p * 4 + 2] / 255;
    chw[p] = (r - MEAN[0]) / STD[0];
    chw[plane + p] = (g - MEAN[1]) / STD[1];
    chw[2 * plane + p] = (b - MEAN[2]) / STD[2];
  }

  const input = new Tensor("float32", chw, [1, 3, S, S]);
  const output = await model({ [inputName]: input });
  const sal = output["1959"] ?? output[Object.keys(output)[0]];
  const data = sal.data;

  const mask = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) {
    const v = data[i] * 255;
    mask[i] = v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v);
  }
  return mask;
}

export const MODEL_INPUT_SIZE = S;
