// Main-thread rendering: draw the composited cut-out over a transparency checkerboard so the
// alpha is *visible*, and encode the full-resolution result as a PNG-with-alpha for download.
import type { RGBAImage } from "./composite.ts";

const CHECKER = 12; // px per checker square

// Our RGBA buffers are always ArrayBuffer-backed; TS 5.7+ generic typed arrays can't prove that
// against ImageData's ArrayBuffer requirement, so narrow at the one construction boundary.
function toImageData(img: RGBAImage): ImageData {
  return new ImageData(img.data as Uint8ClampedArray<ArrayBuffer>, img.width, img.height);
}

/** Draw a checkerboard backdrop, then the RGBA cut-out over it, into the given canvas. */
export function renderPreview(canvas: HTMLCanvasElement, img: RGBAImage): void {
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D context unavailable");

  // Checkerboard so transparent regions read as transparent, not as a flat color.
  for (let y = 0; y < img.height; y += CHECKER) {
    for (let x = 0; x < img.width; x += CHECKER) {
      const dark = ((x / CHECKER) | 0) % 2 === ((y / CHECKER) | 0) % 2;
      ctx.fillStyle = dark ? "#cbcbcb" : "#f4f4f4";
      ctx.fillRect(x, y, CHECKER, CHECKER);
    }
  }

  // Composite the RGBA (with real alpha) over the checkerboard via a scratch canvas + drawImage,
  // so putImageData (which would overwrite, ignoring alpha) doesn't erase the backdrop.
  const scratch = document.createElement("canvas");
  scratch.width = img.width;
  scratch.height = img.height;
  scratch.getContext("2d")!.putImageData(toImageData(img), 0, 0);
  ctx.drawImage(scratch, 0, 0);
}

/** Encode the full-resolution RGBA (with alpha) as a PNG Blob — never downscaled. */
export function toPngBlob(img: RGBAImage): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D context unavailable");
  ctx.putImageData(toImageData(img), 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("toBlob returned null"))), "image/png");
  });
}

/** Trigger a client-side download of a Blob (no server round-trip). */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoke after the click has been processed.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
