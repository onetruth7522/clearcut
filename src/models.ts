// Model descriptor registry + pure mask-extraction math.
//
// Deliberately transformers-free (no `@huggingface/transformers` import) so the data layer is
// node-unit-testable (test/models.test.ts is the normative spec). segment.ts imports transformers
// at top; keeping descriptors and the sigmoid/saliency transform out of that import keeps the test
// clean. See PHASE-CONTRACT.md (Phase 2b / CF-0010) D1 and §6.

export type QualityKey = "fast" | "hq";
export type PreprocessStrategy = "manual-squash" | "auto-processor";
export type OutputStrategy = "saliency" | "logit-sigmoid";

export interface ModelDescriptor {
  key: QualityKey;
  id: string;                 // HF repo id
  dtype: "fp32" | "fp16";
  device: "webgpu" | "auto";  // hq = "webgpu" (no WASM fallback); fast = "auto" (current behavior)
  inputSize: number;          // model input edge in px: fast 320, hq 1024
  preprocess: PreprocessStrategy;
  output: OutputStrategy;
  outputKey?: string;         // fast reads "1959"; hq reads its single output tensor by name
}

// fast = exactly the Phase-1 U-2-Netp constants (the free-path-unchanged anchor at the data layer).
// hq  = BiRefNet_lite fp16 @ 1024², WebGPU-only; id pinned at gate by the §5 precondition smoke
//       (the bare onnx-community/BiRefNet_lite id only redirects to this -ONNX repo).
export const MODELS: Record<QualityKey, ModelDescriptor> = {
  fast: {
    key: "fast",
    id: "BritishWerewolf/U-2-Netp",
    dtype: "fp32",
    device: "auto",
    inputSize: 320,
    preprocess: "manual-squash",
    output: "saliency",
    outputKey: "1959",
  },
  hq: {
    key: "hq",
    id: "onnx-community/BiRefNet_lite-ONNX",
    dtype: "fp16",
    device: "webgpu",
    inputSize: 1024,
    preprocess: "auto-processor",
    output: "logit-sigmoid",
  },
};

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Convert a raw model output plane to a single-channel uint8 mask (0..255, high = foreground).
 * - "saliency": data is already 0..1 sigmoid saliency (U-2-Netp) — scale directly. UNCHANGED from Phase 1.
 * - "logit-sigmoid": data is raw logits (BiRefNet) — sigmoid first, then scale.
 * Both clamp/round identically to the original Phase-1 extraction. §6 is the normative spec.
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
