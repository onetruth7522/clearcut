// Inference worker. Owns the model(s) so loads + segmentation never block the UI thread.
//
// Protocol (main -> worker):
//   { type: "init", ortBase }                       configure env once
//   { type: "segment", id, blob, model }            segment one image with the chosen quality model
// Protocol (worker -> main):
//   { type: "backend", backend }                    active backend resolved (webgpu | wasm)
//   { type: "progress", progress }                  model download/load progress
//   { type: "result", id, mask, maskW, maskH }      transferable square uint8 mask (320² or 1024²)
//   { type: "error", id?, message }
import { configureEnv, ensureModel, segment } from "./segment.ts";
import type { QualityKey } from "./models.ts";

type InMsg =
  | { type: "init"; ortBase: string }
  | { type: "segment"; id: number; blob: Blob; model: QualityKey };

// Report the active backend once per resolved model (fast/hq can resolve to different backends).
const backendReported = new Set<QualityKey>();

self.onmessage = async (e: MessageEvent<InMsg>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      configureEnv(msg.ortBase);
      return;
    }
    if (msg.type === "segment") {
      // Lazy-load the chosen model on first request; surface progress + the chosen backend.
      const backend = await ensureModel(msg.model, (p) => self.postMessage({ type: "progress", progress: p }));
      if (!backendReported.has(msg.model)) {
        backendReported.add(msg.model);
        self.postMessage({ type: "backend", backend });
      }

      // EXIF-decode once here so the mask aligns with the main thread's identically-decoded original.
      // segment() owns the model-specific preprocessing and closes the bitmap.
      const bitmap = await createImageBitmap(msg.blob, { imageOrientation: "from-image" });
      const { data: mask, size } = await segment(msg.model, bitmap);

      self.postMessage(
        { type: "result", id: msg.id, mask, maskW: size, maskH: size },
        { transfer: [mask.buffer] },
      );
    }
  } catch (err) {
    const id = msg.type === "segment" ? msg.id : undefined;
    self.postMessage({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
  }
};
