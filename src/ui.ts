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
  };
}

/** Reflect pro entitlement into the UI: show the PRO badge, retire the unlock entry, allow multi-file intake. */
export function setPro(refs: UIRefs, active: boolean): void {
  refs.proBadge.hidden = !active;
  refs.unlockToggle.hidden = active;
  if (active) refs.unlockPanel.hidden = true;
  // Pro unlocks multi-image batch intake; free stays single-file.
  refs.fileInput.multiple = active;
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
