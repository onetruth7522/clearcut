// UI wiring: intake (pick/drop) -> worker (segment) -> composite -> preview -> download.
// The original full-resolution pixels stay on THIS thread; the worker returns a square mask
// (320² Fast / 1024² HQ), which we upscale and composite at full resolution here.
import { compositeAlpha, type RGBAImage } from "./composite.ts";
import { renderPreview, toPngBlob, downloadBlob } from "./render.ts";
import { getRefs, setStatus, showError, setDragActive, setPro, setUnlockMsg, renderBatch, setQuality, setQualityAffordance, type UIRefs } from "./ui.ts";
import { verifyProToken, verifyStoredEntitlement, loadPro, savePro } from "./license.ts";
import type { QualityKey } from "./models.ts";
import { downloadZip } from "client-zip";

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
let isPro = false; // Pro entitlement (verified offline); unlocks batch + bulk-ZIP. See license.ts.
// Quality model (Phase 2b / CF-0010, Amendment 1). Default High-Quality when Pro. HQ (IS-Net) runs
// on WASM and is available to EVERY Pro user (no WebGPU required) — it's just slower. Free = Fast.
let quality: QualityKey = "hq";

const effectiveModel = (): QualityKey => (isPro ? quality : "fast");

// Quality affordance: the segmented toggle + "slower" hint when Pro; nothing when free.
function updateQualityAffordance(): void {
  setQualityAffordance(refs, isPro);
  if (isPro) setQuality(refs, quality);
}

// Reflect Pro entitlement into UI + state: badge/intake (setPro), default the quality to HQ, refresh
// the affordance. Called from both unlock paths (live activate + stored-token restore).
function enableProUI(): void {
  isPro = true;
  setPro(refs, true);
  quality = "hq";
  updateQualityAffordance();
}

// Graceful revert when an HQ job fails: a failed HQ load/inference (download failure, OOM, or an
// inference error) would otherwise have every subsequent drop re-attempt the same failing path.
// Drop back to Fast and refresh the toggle so the user sees the change (the error itself is already
// surfaced by the caller). Only meaningful while still on HQ.
function revertHqToFast(): void {
  if (quality === "hq") {
    quality = "fast";
    updateQualityAffordance();
  }
}

// Batch queue. Free = a 1-item queue; Pro = N items processed STRICTLY sequentially through the
// single ORT worker session — never two concurrent segmentations (ARCHITECTURE.md §6 invariant).
type JobStatus = "queued" | "processing" | "done" | "error";
interface Job {
  id: number;
  file: File;
  name: string;        // source filename
  status: JobStatus;
  model: QualityKey;   // model frozen at intake — a whole batch runs one model (no mid-batch mix)
  detail?: string;     // error message, when status === "error"
  outName?: string;    // "<stem>-clearcut.png"
  blob?: Blob;         // encoded transparent PNG, retained for bulk-ZIP
}
let jobs: Job[] = [];
let inFlight = false;  // a segmentation is currently in the worker (the single-session guard)

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
    void onResult(msg);
  } else if (msg.type === "error") {
    inFlight = false;
    const job = msg.id !== undefined ? jobs.find((j) => j.id === msg.id) : undefined;
    if (msg.id !== undefined) pending.delete(msg.id); // don't leak the held full-res RGBA
    if (job) { job.status = "error"; job.detail = msg.message; }
    if (job?.model === "hq") revertHqToFast(); // failed HQ path -> drop to Fast (D4)
    if (isPro) { renderQueue(); updateOverallStatus(); } else { showError(refs, msg.message); }
    pump(); // keep the batch moving past a failed image
  }
};
// An uncaught worker crash (e.g. ORT fatal) must not wedge the queue: reset the single-session
// guard, fail the in-flight job, and pump on — otherwise inFlight stays true and intake() silently
// ignores every future drop until reload (Debugger M1; stakes raised by the batch queue).
worker.onerror = (e) => {
  inFlight = false;
  const msg = e.message || "worker crashed";
  const stuck = jobs.find((j) => j.status === "processing");
  if (stuck) { stuck.status = "error"; stuck.detail = msg; }
  if (stuck?.model === "hq") revertHqToFast(); // an HQ crash -> drop to Fast (D4)
  if (isPro) { renderQueue(); updateOverallStatus(); } else { showError(refs, msg); }
  pump();
};

// Composite + encode a finished mask. The worker is freed (pump) BEFORE we encode, so the next
// image segments on the worker thread while this one's PNG encodes on the main thread.
async function onResult(msg: WorkerResult): Promise<void> {
  inFlight = false;
  const job = jobs.find((j) => j.id === msg.id);
  const meta = pending.get(msg.id);
  pending.delete(msg.id);
  pump();
  if (!job || !meta) return;
  try {
    const composited = compositeAlpha(meta.original, { data: msg.mask, width: msg.maskW, height: msg.maskH });
    lastResult = composited;
    renderPreview(refs.canvas, composited);
    refs.preview.hidden = false;
    job.outName = stem(job.name) + "-clearcut.png";
    refs.downloadBtn.dataset.name = job.outName;
    refs.downloadBtn.disabled = false;
    job.blob = await toPngBlob(composited); // retain for bulk-ZIP (hold the PNG, not the RGBA)
    job.status = "done";
    if (!isPro) setStatus(refs, "done", activeBackend ? `via ${activeBackend}` : "");
  } catch (err) {
    job.status = "error";
    job.detail = err instanceof Error ? err.message : String(err);
    if (!isPro) showError(refs, job.detail);
  }
  if (isPro) { renderQueue(); updateOverallStatus(); }
  syncDownloadAll();
}

const stem = (filename: string): string => filename.replace(/\.[^.]+$/, "");

// Decode a file to FULL-RESOLUTION RGBA on this thread; these pixels are kept for the composite.
// Honor EXIF orientation (phone JPEGs) — must match the worker's decode so the mask aligns.
async function decodeFullRes(file: File): Promise<RGBAImage> {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  if (bitmap.width * bitmap.height > MAX_PIXELS) {
    bitmap.close();
    throw new Error(`image too large (${(bitmap.width * bitmap.height / 1e6).toFixed(0)} MP) — max 25 MP`);
  }
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) { bitmap.close(); throw new Error("2D context unavailable"); }
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return {
    data: ctx.getImageData(0, 0, canvas.width, canvas.height).data,
    width: canvas.width,
    height: canvas.height,
  };
}

function queueBusy(): boolean {
  return inFlight || jobs.some((j) => j.status === "queued");
}

// Accept dropped/picked files into a FRESH queue and start the pump. Free intake is capped to the
// first accepted file; Pro enqueues every accepted image.
function intake(files: File[]): void {
  if (queueBusy()) return; // a batch is already running — ignore new drops (single ORT session)
  const accepted = files.filter((f) => ACCEPTED.has(f.type));
  if (accepted.length === 0) {
    showError(refs, "unsupported file type — use PNG, JPEG, or WebP");
    return;
  }
  const batch = isPro ? accepted : accepted.slice(0, 1);
  // Freeze the model for the whole batch at intake — toggling mid-batch must not mix models.
  const batchModel = effectiveModel();
  jobs = batch.map((file) => ({ id: ++reqId, file, name: file.name, status: "queued", model: batchModel }));
  refs.downloadBtn.disabled = true;
  refs.preview.hidden = true;
  setStatus(refs, "loading-model");
  renderQueue();
  syncDownloadAll(); // fresh batch — nothing to zip yet
  pump();
}

// Dispatch the next queued job into the worker. The inFlight guard ensures only ONE segmentation
// is ever outstanding — we never postMessage a second segment until the first's result arrives.
function pump(): void {
  if (inFlight) return;
  const next = jobs.find((j) => j.status === "queued");
  if (!next) return; // queue drained
  inFlight = true;
  next.status = "processing";
  renderQueue();
  void dispatch(next);
}

async function dispatch(job: Job): Promise<void> {
  try {
    const original = await decodeFullRes(job.file);
    pending.set(job.id, { original, sourceName: job.name });
    if (isPro) updateOverallStatus(); else setStatus(refs, "segmenting");
    // Hand the worker the original blob (cheap to clone) + the job's frozen quality model; the worker
    // EXIF-decodes and the model owner does its own per-model downscale/preprocess.
    worker.postMessage({ type: "segment", id: job.id, blob: job.file, model: job.model });
  } catch (err) {
    // Decode failed (e.g. too large / corrupt) — fail just this job and keep the batch moving.
    job.status = "error";
    job.detail = err instanceof Error ? err.message : String(err);
    inFlight = false;
    if (!isPro) showError(refs, job.detail);
    renderQueue();
    updateOverallStatus();
    pump();
  }
}

function renderQueue(): void {
  // Free single-image flow keeps the classic status line and shows no list; the list is Pro-only.
  renderBatch(refs, isPro ? jobs.map((j) => ({ name: j.name, status: j.status, detail: j.detail })) : []);
}

function doneCount(): number {
  return jobs.filter((j) => j.status === "done" || j.status === "error").length;
}

function updateOverallStatus(): void {
  if (!isPro) return; // the free flow's status is set inline by the single-job path
  const total = jobs.length;
  if (total === 0) return;
  const done = doneCount();
  if (done >= total) {
    const errs = jobs.filter((j) => j.status === "error").length;
    const detail = `${total - errs} of ${total} done${errs ? `, ${errs} failed` : ""}` +
      (activeBackend ? ` — via ${activeBackend}` : "");
    setStatus(refs, "done", detail);
  } else {
    setStatus(refs, "segmenting", `${done} of ${total} done`);
  }
}

// Finished images with an encoded PNG, in queue order — the bulk-ZIP payload. De-duplicate output
// names so two same-named sources (e.g. batched from different folders) don't overwrite each other
// inside the ZIP (Debugger L3): the Nth collision becomes "<stem> (N)-clearcut.png".
function readyResults(): { name: string; blob: Blob }[] {
  const seen = new Map<string, number>();
  return jobs.flatMap((j) => {
    if (!(j.status === "done" && j.blob && j.outName)) return [];
    const n = seen.get(j.outName) ?? 0;
    seen.set(j.outName, n + 1);
    let name = j.outName;
    if (n > 0) {
      const dot = name.lastIndexOf(".");
      name = dot === -1 ? `${name} (${n})` : `${name.slice(0, dot)} (${n})${name.slice(dot)}`;
    }
    return [{ name, blob: j.blob }];
  });
}

function syncDownloadAll(): void {
  if (!isPro) return;
  refs.downloadAllBtn.disabled = readyResults().length === 0;
}

// --- intake wiring ---------------------------------------------------------
refs.fileInput.addEventListener("change", () => {
  const files = refs.fileInput.files;
  if (files && files.length) intake([...files]);
  refs.fileInput.value = ""; // allow re-picking the same file(s)
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
    const files = e.dataTransfer?.files;
    if (files && files.length) intake([...files]);
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

// Bulk-ZIP (Pro): stream all finished PNGs into one store-only ZIP, entirely in-browser.
refs.downloadAllBtn.addEventListener("click", async () => {
  const ready = readyResults();
  if (ready.length === 0) return;
  refs.downloadAllBtn.disabled = true;
  try {
    // client-zip stores (never deflates) — ideal for already-compressed PNGs — and streams,
    // so a large batch zips without buffering every file in memory at once.
    const blob = await downloadZip(ready.map((r) => ({ name: r.name, input: r.blob }))).blob();
    downloadBlob(blob, "clearcut-batch.zip");
  } catch (err) {
    showError(refs, `could not build ZIP: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    syncDownloadAll();
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
    enableProUI();
    savePro(token);
    setUnlockMsg(refs, "Pro unlocked — batch, bulk-ZIP, and the High-Quality model are enabled.", "ok");
  } else {
    setUnlockMsg(refs, "That token isn't valid. Check you pasted the whole thing.", "error");
  }
}
refs.activateBtn.addEventListener("click", () => void activate());
refs.tokenInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); void activate(); }
});

// Quality toggle (Pro only): sets the model for the NEXT batch (each batch freezes its model at
// intake). Event-delegated so one listener covers both segments; a running batch is unaffected.
refs.qualityToggle.addEventListener("click", (e) => {
  const btn = (e.target as HTMLElement).closest<HTMLButtonElement>(".q-opt");
  const key = btn?.dataset.quality;
  if (key !== "fast" && key !== "hq") return;
  quality = key;
  setQuality(refs, quality);
});

// Restore entitlement on load: re-verify the stored token (a hand-set flag never unlocks).
void verifyStoredEntitlement(loadPro()).then((ok) => {
  if (ok) enableProUI();
});

setStatus(refs, "idle");
