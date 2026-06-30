// Edge defringe / halo decontamination (CF-0011).
//
// compositeAlpha preserves the camera's ORIGINAL RGB at every pixel and writes only alpha. At the
// subject boundary the upscaled mask is fractional, but those edge pixels' RGB is the foreground
// already optically blended with the (often dark, sometimes light) background — α·FG + (1−α)·BG.
// Stored straight-alpha, that contaminated color re-emerges as a colored rim/halo when the cut-out is
// composited onto a new background (most visible over a contrasting one — e.g. a light-bg photo shows a
// pale rim over black).
//
// This module decontaminates edge color by FOREGROUND-COLOR PROPAGATION: it floods each fully-opaque
// "solid" foreground color outward along the matte band (a multi-source breadth-first grassfire), so
// every fractional-alpha "fringe" pixel takes the color of its NEAREST solid foreground pixel. The
// ALPHA CHANNEL IS NEVER MODIFIED, so the subject is not eroded (the no-erosion guarantee).
//
// Why propagation, not a fixed-radius window (CF-0011 Amendment 1): the free U²-Netp model emits a soft
// matte whose anti-aliased band is wide and variable — measured median ~39 px, p90 ~81 px, up to ~375 px
// on the cold-drive photo. A bounded search window cannot reach a solid source across a band that wide
// (and a window large enough would be billions of ops), so the outer band keeps its contaminated color
// and the halo survives. BFS reaches any distance in O(W·H). See HISTORY / PHASE-CONTRACT Amendment 1.
//
// Pure typed-array math, no DOM — unit-testable in node (test/defringe.test.ts is the normative spec).

import type { RGBAImage } from "./composite.ts";

// A pixel is a clean color SOURCE when its alpha is at/above this — interior foreground. Pixels with
// alpha in (0, SOLID_T) are contaminated "fringe" edge pixels whose RGB we decontaminate; alpha 0 is
// fully transparent (never shown), so it is left untouched. One threshold: every partial-alpha pixel is
// fringe, every near/fully-opaque pixel is a source.
const SOLID_T = 250;

/**
 * Decontaminate edge colors in a composited RGBA image. Returns a NEW image with the SAME dimensions
 * and a BIT-IDENTICAL alpha channel; only the RGB of fractional-alpha edge pixels is changed, set to
 * the color of the nearest fully-opaque foreground pixel (propagated outward through the matte band).
 * Solid and fully-transparent pixels keep their RGB. A fringe pixel with no solid pixel reachable
 * through the band keeps its original RGB (safe, bounded fallback).
 */
export function defringe(img: RGBAImage): RGBAImage {
  const W = img.width;
  const H = img.height;
  const src = img.data;
  const N = W * H;
  const out = new Uint8ClampedArray(src.length);
  out.set(src); // alpha + all non-filled RGB carried through unchanged

  // Multi-source BFS (grassfire). Seed every solid pixel; flood its color outward to fringe neighbors,
  // which in turn seed their fringe neighbors — so color walks the whole matte ramp from the solid core
  // out to the faintest edge. We propagate ONLY into fringe pixels (not transparent), confining the
  // sweep to the subject + its band, and each filled pixel inherits the color that reached it (nearest
  // solid). Reading neighbor colors from `out` is safe because a pixel's color is final once `known`.
  const known = new Uint8Array(N);
  const queue = new Int32Array(N);
  let head = 0, tail = 0;
  for (let p = 0; p < N; p++) {
    if (src[p * 4 + 3] >= SOLID_T) { known[p] = 1; queue[tail++] = p; }
  }

  while (head < tail) {
    const p = queue[head++];
    const o = p * 4;
    const r = out[o], g = out[o + 1], b = out[o + 2];
    const x = p % W;
    const y = (p - x) / W;
    // 4-connected neighbors; fill each fringe neighbor with this pixel's (nearest-solid) color.
    if (x > 0) {
      const n = p - 1;
      if (!known[n]) { const a = src[n * 4 + 3]; if (a !== 0 && a < SOLID_T) { const no = n * 4; out[no] = r; out[no + 1] = g; out[no + 2] = b; known[n] = 1; queue[tail++] = n; } }
    }
    if (x < W - 1) {
      const n = p + 1;
      if (!known[n]) { const a = src[n * 4 + 3]; if (a !== 0 && a < SOLID_T) { const no = n * 4; out[no] = r; out[no + 1] = g; out[no + 2] = b; known[n] = 1; queue[tail++] = n; } }
    }
    if (y > 0) {
      const n = p - W;
      if (!known[n]) { const a = src[n * 4 + 3]; if (a !== 0 && a < SOLID_T) { const no = n * 4; out[no] = r; out[no + 1] = g; out[no + 2] = b; known[n] = 1; queue[tail++] = n; } }
    }
    if (y < H - 1) {
      const n = p + W;
      if (!known[n]) { const a = src[n * 4 + 3]; if (a !== 0 && a < SOLID_T) { const no = n * 4; out[no] = r; out[no + 1] = g; out[no + 2] = b; known[n] = 1; queue[tail++] = n; } }
    }
  }

  return { data: out, width: W, height: H };
}
