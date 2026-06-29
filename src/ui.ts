// Small DOM helpers: status line, dropzone state, error surface. No app logic here.

export type Status = "idle" | "loading-model" | "segmenting" | "done" | "error";

export interface UIRefs {
  dropzone: HTMLElement;
  fileInput: HTMLInputElement;
  status: HTMLElement;
  canvas: HTMLCanvasElement;
  downloadBtn: HTMLButtonElement;
  preview: HTMLElement;
  // Pro unlock (Phase 2)
  unlockToggle: HTMLElement;
  unlockPanel: HTMLElement;
  tokenInput: HTMLInputElement;
  activateBtn: HTMLButtonElement;
  unlockMsg: HTMLElement;
  proBadge: HTMLElement;
  batchList: HTMLElement;
  downloadAllBtn: HTMLButtonElement;
  // Pro-only quality toggle (Phase 2b)
  qualityRow: HTMLElement;
  qualityToggle: HTMLElement;
  qualityNote: HTMLElement;
}

export function getRefs(): UIRefs {
  const $ = <T extends HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`missing #${id}`);
    return el as T;
  };
  return {
    dropzone: $("dropzone"),
    fileInput: $<HTMLInputElement>("file"),
    status: $("status"),
    canvas: $<HTMLCanvasElement>("canvas"),
    downloadBtn: $<HTMLButtonElement>("download"),
    preview: $("preview"),
    unlockToggle: $("unlock-toggle"),
    unlockPanel: $("unlock-panel"),
    tokenInput: $<HTMLInputElement>("token-input"),
    activateBtn: $<HTMLButtonElement>("activate"),
    unlockMsg: $("unlock-msg"),
    proBadge: $("pro-badge"),
    batchList: $("batch"),
    downloadAllBtn: $<HTMLButtonElement>("download-all"),
    qualityRow: $("quality-row"),
    qualityToggle: $("quality-toggle"),
    qualityNote: $("quality-note"),
  };
}

/**
 * Show the quality affordance in one of three states (decided by main.ts):
 *   "toggle" — Pro + WebGPU: the Fast / High-Quality switch
 *   "note"   — Pro, no WebGPU: the "High-Quality needs a WebGPU browser" note
 *   "none"   — free: nothing
 */
export function setQualityAffordance(refs: UIRefs, mode: "toggle" | "note" | "none"): void {
  refs.qualityRow.hidden = mode === "none";
  refs.qualityToggle.hidden = mode !== "toggle";
  refs.qualityNote.hidden = mode !== "note";
}

/** Reflect the selected quality on the segmented control (aria-checked drives the visual state). */
export function setQuality(refs: UIRefs, key: "fast" | "hq"): void {
  for (const btn of refs.qualityToggle.querySelectorAll<HTMLButtonElement>(".q-opt")) {
    btn.setAttribute("aria-checked", String(btn.dataset.quality === key));
  }
}

export type BatchStatus = "queued" | "processing" | "done" | "error";
export interface BatchItemView {
  name: string;
  status: BatchStatus;
  detail?: string;
}

/** Render the Pro batch queue as a per-image status list. An empty list hides the container. */
export function renderBatch(refs: UIRefs, items: BatchItemView[]): void {
  const list = refs.batchList;
  if (items.length === 0) {
    list.hidden = true;
    list.replaceChildren();
    return;
  }
  const glyph: Record<BatchStatus, string> = { queued: "•", processing: "…", done: "✓", error: "✗" };
  list.hidden = false;
  list.replaceChildren(
    ...items.map((it) => {
      const row = document.createElement("div");
      row.className = "batch-item";
      row.dataset.status = it.status;
      const name = document.createElement("span");
      name.className = "bi-name";
      name.textContent = it.name;
      const state = document.createElement("span");
      state.className = "bi-state";
      state.textContent = it.detail ? `${glyph[it.status]} ${it.detail}` : glyph[it.status];
      row.append(name, state);
      return row;
    }),
  );
}

/** Reflect pro entitlement into the UI: show the PRO badge, retire the unlock entry, allow multi-file intake. */
export function setPro(refs: UIRefs, active: boolean): void {
  refs.proBadge.hidden = !active;
  refs.unlockToggle.hidden = active;
  if (active) refs.unlockPanel.hidden = true;
  // Pro unlocks multi-image batch intake; free stays single-file.
  refs.fileInput.multiple = active;
  // Reveal the bulk-ZIP button for Pro (it stays disabled until ≥1 result exists).
  refs.downloadAllBtn.hidden = !active;
}

/** Inline feedback under the token box. `state` "ok" | "error" colors it; "" clears. */
export function setUnlockMsg(refs: UIRefs, message: string, state: "ok" | "error" | "" = ""): void {
  refs.unlockMsg.textContent = message;
  if (state) refs.unlockMsg.dataset.state = state;
  else delete refs.unlockMsg.dataset.state;
}

export function setStatus(refs: UIRefs, status: Status, detail = ""): void {
  refs.status.dataset.state = status;
  const labels: Record<Status, string> = {
    "idle": "Drop an image to remove its background",
    "loading-model": "Loading model",
    "segmenting": "Removing background",
    "done": "Done — your photo never left this device",
    "error": "Something went wrong",
  };
  refs.status.textContent = detail ? `${labels[status]} — ${detail}` : labels[status];
}

export function showError(refs: UIRefs, message: string): void {
  setStatus(refs, "error", message);
}

export function setDragActive(refs: UIRefs, active: boolean): void {
  refs.dropzone.classList.toggle("active", active);
}
