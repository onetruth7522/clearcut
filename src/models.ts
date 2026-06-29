// Model descriptor registry + pure mask-extraction math.
//
// Deliberately transformers/ort-free (no runtime import) so the data layer is node-unit-testable
// (test/models.test.ts is the normative spec). See PHASE-CONTRACT.md (Phase 2b / CF-0010, Amendment 1).
//
// Two models, two runtimes:
//   fast (U-2-Netp, 320²)        — runtime "transformers" (@huggingface/transformers AutoModel),
//                                   manual ImageNet preprocess, output "1959" (already 0..1).
//   hq   (IS-Net general-use,    — runtime "ort-raw" (raw onnxruntime-web session on a bare Apache
//        1024²)                     .onnx; transformers.js can't load it — no HF config). WASM only
//                                   (the Apache export's MaxPool ceil_mode is unsupported on ORT-Web
//                                   WebGPU). Output "output" (already 0..1). ~10s/image; runs everywhere.
// Preprocessing is unified + descriptor-driven (inputSize, rescale, mean, std) for both; the only
// difference is the inference call (transformers model() vs ort session.run()).

export type QualityKey = "fast" | "hq";
export type Runtime = "transformers" | "ort-raw";
export type OutputStrategy = "saliency" | "logit-sigmoid"; // "saliency" = output already 0..1

export interface ModelDescriptor {
  key: QualityKey;
  runtime: Runtime;
  inputSize: number;              // model input edge in px: fast 320, hq 1024
  rescale: boolean;               // divide pixels by 255 before normalize (U-2-Netp yes; IS-Net no)
  mean: [number, number, number]; // per-channel normalize mean
  std: [number, number, number];  // per-channel normalize std
  output: OutputStrategy;

  // runtime "transformers"
  id?: string;                    // HF repo id
  dtype?: "fp32" | "fp16";
  device?: "webgpu" | "auto";
  outputKey?: string;             // tensor key to read (U-2-Netp "1959")

  // runtime "ort-raw"
  url?: string;                   // direct .onnx URL (fetched at runtime, zero-backend)
  ep?: "wasm" | "webgpu";         // execution provider
  inputName?: string;             // ONNX input name
  outputName?: string;            // ONNX output name to read
}

export const MODELS: Record<QualityKey, ModelDescriptor> = {
  // EXACTLY the Phase-1 U-2-Netp behavior (the free-path-unchanged anchor at the data layer).
  fast: {
    key: "fast",
    runtime: "transformers",
    inputSize: 320,
    rescale: true,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    output: "saliency",
    id: "BritishWerewolf/U-2-Netp",
    dtype: "fp32",
    device: "auto",
    outputKey: "1959",
  },
  // IS-Net general-use, Apache-2.0 bare ONNX (no HF config) -> raw onnxruntime-web, WASM only.
  // Probed I/O: input "input", 1024², (pixel-128)/256 (do_rescale:false), output "output" 0..1.
  hq: {
    key: "hq",
    runtime: "ort-raw",
    inputSize: 1024,
    rescale: false,
    mean: [128, 128, 128],
    std: [256, 256, 256],
    output: "saliency",
    // Mirror we control (CF-0016): IS-Net general-use, Apache-2.0, attribution in the repo card.
    // Served as uint8 dynamic-quantized q8 (CF-0017): ~44 MB vs 176 MB fp32 (4x smaller), ~1.8x faster
    // on the WASM CPU EP, mask quality near-identical (IoU ~0.94 on a fine-edge test, fine detail kept).
    // fp16 was evaluated and rejected — ORT-Web WASM has no fp16 compute kernels (2x only, equal/slower);
    // q8 dominates on our WASM-only Pro path. SHA-256 feed6f32a5e707ca7e939576b2d891b23fb9eb4114749657a5efc64e8651e43a.
    url: "https://huggingface.co/SacredNoir/isnet-general-use-onnx/resolve/main/isnet-general-use-q8.onnx",
    ep: "wasm",
    inputName: "input",
    outputName: "output",
  },
};

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Build a normalized NCHW float32 tensor [1,3,size,size] from a row-major RGBA buffer (size²·4).
 * Descriptor-driven: `rescale` ÷255 first (U-2-Netp), else raw 0..255 (IS-Net); then (v-mean)/std.
 */
export function preprocessNCHW(
  rgba: Uint8ClampedArray | Uint8Array,
  size: number,
  rescale: boolean,
  mean: [number, number, number],
  std: [number, number, number],
): Float32Array {
  const plane = size * size;
  const chw = new Float32Array(3 * plane);
  const scale = rescale ? 1 / 255 : 1;
  for (let p = 0; p < plane; p++) {
    const r = rgba[p * 4] * scale;
    const g = rgba[p * 4 + 1] * scale;
    const b = rgba[p * 4 + 2] * scale;
    chw[p] = (r - mean[0]) / std[0];
    chw[plane + p] = (g - mean[1]) / std[1];
    chw[2 * plane + p] = (b - mean[2]) / std[2];
  }
  return chw;
}

/**
 * Convert a raw model output plane to a single-channel uint8 mask (0..255, high = foreground).
 * - "saliency": data already 0..1 — scale directly. (Both U-2-Netp "1959" and IS-Net "output".)
 * - "logit-sigmoid": data is raw logits — sigmoid first. (Currently unused; kept + tested.)
 * Clamp/round identical to the original Phase-1 extraction. §6 is the normative spec.
 */
export function maskFromOutput(output: OutputStrategy, data: Float32Array | number[]): Uint8Array {
  const n = data.length;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const s = output === "saliency" ? data[i] : sigmoid(data[i]);
    const v = s * 255;
    mask[i] = v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v);
  }
  return mask;
}
