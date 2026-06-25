// Inference worker. Owns the model so the ~4.4MB load + segmentation never block the UI thread.
//
// Protocol (main -> worker):
//   { type: "init", ortBase }                 configure env once
//   { type: "segment", id, blob }              segment one image (original full-res blob)
// Protocol (worker -> main):
//   { type: "backend", backend }              active backend resolved (webgpu | wasm)
//   { type: "progress", progress }            model download/load progress
//   { type: "result", id, mask, maskW, maskH }  transferable 320x320 uint8 mask
//   { type: "error", id?, message }
import { configureEnv, ensureModel, segment, MODEL_INPUT_SIZE } from "./segment.ts";

type InMsg =
  | { type: "init"; ortBase: string }
  | { type: "segment"; id: number; blob: Blob };

let backendReported = false;

self.onmessage = async (e: MessageEvent<InMsg>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      configureEnv(msg.ortBase);
      return;
    }
    if (msg.type === "segment") {
      // Lazy-load the model on first real request; surface progress + the chosen backend.
      const backend = await ensureModel((p) => self.postMessage({ type: "progress", progress: p }));
      if (!backendReported) {
        backendReported = true;
        self.postMessage({ type: "backend", backend });
      }

      // Squash-resize the original to the model's 320x320 input (canonical U2Net preprocessing).
      // Honor EXIF orientation so the mask aligns with the main thread's identically-decoded original.
      const bitmap = await createImageBitmap(msg.blob, { imageOrientation: "from-image" });
      const S = MODEL_INPUT_SIZE;
      const canvas = new OffscreenCanvas(S, S);
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) throw new Error("OffscreenCanvas 2D context unavailable in worker");
      ctx.drawImage(bitmap, 0, 0, S, S);
      bitmap.close();
      const rgba = ctx.getImageData(0, 0, S, S).data;

      const mask = await segment(rgba);
      self.postMessage(
        { type: "result", id: msg.id, mask, maskW: S, maskH: S },
        { transfer: [mask.buffer] },
      );
    }
  } catch (err) {
    const id = msg.type === "segment" ? msg.id : undefined;
    self.postMessage({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
  }
};
