// Edge defringe / halo decontamination (CF-0011).
//
// compositeAlpha preserves the camera's ORIGINAL RGB at every pixel and writes only alpha. At the
// subject boundary the upscaled mask is fractional, but those edge pixels' RGB is the foreground
// already optically blended with the (often dark) background — α·FG + (1−α)·BG. Stored straight-alpha,
// that contaminated color re-emerges as a dark rim/halo when the cut-out lands on a new background.
//
// This module decontaminates edge color: for each fractional-alpha "fringe" pixel it replaces the RGB
// with a distance-weighted average of nearby fully-opaque "solid" foreground pixels — pushing clean
// interior color outward to the edge. The ALPHA CHANNEL IS NEVER MODIFIED, so the subject is not
// eroded (the no-erosion guarantee). Sources are read from the INPUT buffer only, so the result is
// order-independent. Pure typed-array math, no DOM — unit-testable in node (test/defringe.test.ts is
// the normative spec).

import type { RGBAImage } from "./composite.ts";

// A pixel is a clean color SOURCE when its alpha is at/above this — interior foreground. Pixels with
// alpha in (0, SOLID_T) are contaminated edge pixels whose RGB we decontaminate; alpha 0 is fully
// transparent (never shown), so it is left untouched.
const SOLID_T = 250;
// Search radius (px) around a fringe pixel for solid sources. Bounded so cost stays O(W·H·R²).
const R = 4;

/**
 * Decontaminate edge colors in a composited RGBA image. Returns a NEW image with the SAME dimensions
 * and a BIT-IDENTICAL alpha channel; only the RGB of fractional-alpha edge pixels is changed, pulled
 * toward nearby fully-opaque foreground color. Solid and fully-transparent pixels keep their RGB.
 */
export function defringe(img: RGBAImage): RGBAImage {
  const W = img.width;
  const H = img.height;
  const src = img.data;
  const out = new Uint8ClampedArray(src.length);
  out.set(src); // start as a copy: alpha + all non-fringe RGB carried through unchanged

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const a = src[i + 3];
      if (a === 0 || a >= SOLID_T) continue; // transparent or solid — RGB already copied as-is

      // Fringe pixel: gather distance-weighted clean color from solid neighbors in the INPUT buffer.
      let wr = 0, wg = 0, wb = 0, wsum = 0;
      const y0 = y - R < 0 ? 0 : y - R;
      const y1 = y + R >= H ? H - 1 : y + R;
      const x0 = x - R < 0 ? 0 : x - R;
      const x1 = x + R >= W ? W - 1 : x + R;
      for (let ny = y0; ny <= y1; ny++) {
        for (let nx = x0; nx <= x1; nx++) {
          const j = (ny * W + nx) * 4;
          if (src[j + 3] < SOLID_T) continue; // only pull from solid foreground sources
          const dx = nx - x, dy = ny - y;
          const w = 1 / (dx * dx + dy * dy); // inverse-squared distance; center is fringe, never qualifies
          wr += src[j] * w;
          wg += src[j + 1] * w;
          wb += src[j + 2] * w;
          wsum += w;
        }
      }
      if (wsum > 0) {
        out[i] = Math.round(wr / wsum);
        out[i + 1] = Math.round(wg / wsum);
        out[i + 2] = Math.round(wb / wsum);
        // out[i + 3] is the copied original alpha — NEVER modified (the no-erosion guarantee).
      }
      // else: no solid neighbor in range — leave the copied original RGB (safe, bounded fallback).
    }
  }

  return { data: out, width: W, height: H };
}
