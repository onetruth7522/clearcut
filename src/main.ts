// UI wiring: intake (pick/drop) -> worker (segment) -> composite -> preview -> download.
// The original full-resolution pixels stay on THIS thread; the worker only ever returns a
// 320x320 mask, which we upscale and composite at full resolution here.
import { compositeAlpha, type RGBAImage } from "./composite.ts";
import { renderPreview, toPngBlob, downloadBlob } from "./render.ts";
import { getRefs, setStatus, showError, setDragActive, setPro, setUnlockMsg, type UIRefs } from "./ui.ts";
import { verifyProToken, verifyStoredEntitlement, loadPro, savePro } from "./license.ts";

const ACCEPTED = new Set(["image/png", "image/jpeg", "image/webp"]);

interface WorkerResult {
  type: "result";
  id: number;
  mask: Uint8Array;
  maskW: number;
  maskH: number;
}
type WorkerMsg =
  | { type: "backend"; backend: string }
  | { type: "progress"; progress: unknown }
  | WorkerResult
  | { type: "error"; id?: number; message: string };

const refs = getRefs();

// Spin up the inference worker and point it at our self-hosted ORT wasm (dist/ort/).
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const ortBase = new URL("ort/", document.baseURI).href;
worker.postMessage({ type: "init", ortBase });

let reqId = 0;
const pending = new Map<number, { original: RGBAImage; sourceName: string }>();
let activeBackend = "";
let lastResult: RGBAImage | null = null;
let processing = false; // Phase 1 handles one image at a time; ignore drops while one is in flight.
let isPro = false; // Pro entitlement (verified offline); unlocks batch + bulk-ZIP. See license.ts.

// Guard against megapixel inputs that blow past canvas / typed-array limits (silent blank output).
const MAX_PIXELS = 25_000_000; // ~25 MP

worker.onmessage = (e: MessageEvent<WorkerMsg>) => {
  const msg = e.data;
  if (msg.type === "backend") {
    activeBackend = msg.backend;
  } else if (msg.type === "progress") {
    const p = msg.progress as { status?: string; progress?: number };
    if (p?.status === "progress" && typeof p.progress === "number") {
      setStatus(refs, "loading-model", `${Math.round(p.progress)}%`);
    } else {
      setStatus(refs, "loading-model");
    }
  } else if (msg.type === "result") {
    processing = false;
    const job = pending.get(msg.id);
    pending.delete(msg.id);
    if (!job) return;
    try {
      const composited = compositeAlpha(job.original, { data: msg.mask, width: msg.maskW, height: msg.maskH });
      lastResult = composited;
      renderPreview(refs.canvas, composited);
      refs.preview.hidden = false;
      refs.downloadBtn.disabled = false;
      refs.downloadBtn.dataset.name = job.sourceName.replace(/\.[^.]+$/, "") + "-clearcut.png";
      setStatus(refs, "done", activeBackend ? `via ${activeBackend}` : "");
    } catch (err) {
      showError(refs, err instanceof Error ? err.message : String(err));
    }
  } else if (msg.type === "error") {
    processing = false;
    if (msg.id !== undefined) pending.delete(msg.id); // don't leak the held full-res RGBA
    showError(refs, msg.message);
  }
};
worker.onerror = (e) => showError(refs, e.message || "worker crashed");

async function handleFile(file: File): Promise<void> {
  if (processing) return; // one image at a time (Phase 1); ORT can't run concurrent sessions
  if (!ACCEPTED.has(file.type)) {
    showError(refs, `unsupported file type "${file.type || "unknown"}" — use PNG, JPEG, or WebP`);
    return;
  }
  processing = true;
  try {
    setStatus(refs, "loading-model");
    refs.downloadBtn.disabled = true;
    refs.preview.hidden = true;

    // Decode to FULL-RESOLUTION RGBA on this thread; these pixels are kept for the composite.
    // Honor EXIF orientation (phone JPEGs) — must match the worker's decode so the mask aligns.
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    if (bitmap.width * bitmap.height > MAX_PIXELS) {
      bitmap.close();
      processing = false;
      showError(refs, `image too large (${(bitmap.width * bitmap.height / 1e6).toFixed(0)} MP) — max 25 MP`);
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("2D context unavailable");
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const original: RGBAImage = {
      data: ctx.getImageData(0, 0, canvas.width, canvas.height).data,
      width: canvas.width,
      height: canvas.height,
    };

    const id = ++reqId;
    pending.set(id, { original, sourceName: file.name });
    setStatus(refs, "segmenting");
    // Hand the worker the original blob (cheap to clone); it does its own 320x320 downscale.
    worker.postMessage({ type: "segment", id, blob: file });
  } catch (err) {
    processing = false;
    showError(refs, err instanceof Error ? err.message : String(err));
  }
}

// --- intake wiring ---------------------------------------------------------
refs.fileInput.addEventListener("change", () => {
  const f = refs.fileInput.files?.[0];
  if (f) void handleFile(f);
});

function wireDropzone(refs: UIRefs): void {
  const dz = refs.dropzone;
  dz.addEventListener("click", () => refs.fileInput.click());
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    setDragActive(refs, true);
  });
  dz.addEventListener("dragleave", () => setDragActive(refs, false));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    setDragActive(refs, false);
    const f = e.dataTransfer?.files?.[0];
    if (f) void handleFile(f);
  });
}
wireDropzone(refs);

refs.downloadBtn.addEventListener("click", async () => {
  if (!lastResult) return;
  try {
    const blob = await toPngBlob(lastResult);
    downloadBlob(blob, refs.downloadBtn.dataset.name || "clearcut.png");
  } catch (err) {
    showError(refs, `could not encode PNG: ${err instanceof Error ? err.message : String(err)}`);
  }
});

// --- pro unlock (Phase 2) --------------------------------------------------
refs.unlockToggle.addEventListener("click", () => {
  refs.unlockPanel.hidden = !refs.unlockPanel.hidden;
  if (!refs.unlockPanel.hidden) refs.tokenInput.focus();
});

async function activate(): Promise<void> {
  if (isPro) { setUnlockMsg(refs, "Pro is already unlocked on this device.", "ok"); return; }
  const token = refs.tokenInput.value.trim();
  if (!token) { setUnlockMsg(refs, "Paste your license token first.", "error"); return; }
  refs.activateBtn.disabled = true;
  setUnlockMsg(refs, "Checking…");
  // Verify fully offline against the embedded public key — no network call (Model B).
  const ok = await verifyProToken(token);
  refs.activateBtn.disabled = false;
  if (ok) {
    isPro = true;
    savePro(token);
    setPro(refs, true);
    setUnlockMsg(refs, "Pro unlocked — batch processing and bulk-ZIP are enabled.", "ok");
  } else {
    setUnlockMsg(refs, "That token isn't valid. Check you pasted the whole thing.", "error");
  }
}
refs.activateBtn.addEventListener("click", () => void activate());
refs.tokenInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); void activate(); }
});

// Restore entitlement on load: re-verify the stored token (a hand-set flag never unlocks).
void verifyStoredEntitlement(loadPro()).then((ok) => {
  if (ok) { isPro = true; setPro(refs, true); }
});

setStatus(refs, "idle");
