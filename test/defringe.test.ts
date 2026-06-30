// Normative spec for edge defringe / halo decontamination (CF-0011, Phase 3).
// These examples WIN over any prose in the contract or module.
import { test } from "node:test";
import assert from "node:assert/strict";
import { defringe } from "../src/defringe.ts";
import type { RGBAImage } from "../src/composite.ts";

// Build an RGBAImage from a per-pixel [r,g,b,a] generator.
function image(w: number, h: number, at: (x: number, y: number) => [number, number, number, number]): RGBAImage {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const [r, g, b, a] = at(x, y);
      data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = a;
    }
  }
  return { data, width: w, height: h };
}

test("alpha plane is BIT-IDENTICAL in->out (the no-erosion guarantee)", () => {
  // Mixed alphas: transparent / fringe / solid in a deterministic pattern.
  const alphas = [0, 64, 128, 200, 250, 255];
  const img = image(6, 6, (x, y) => [x * 40, y * 40, 100, alphas[(x + y) % alphas.length]]);
  const out = defringe(img);
  for (let i = 0; i < img.width * img.height; i++) {
    assert.equal(out.data[i * 4 + 3], img.data[i * 4 + 3], `alpha at pixel ${i} unchanged`);
  }
});

test("all-opaque image is a no-op on RGB (no fringe -> nothing to decontaminate)", () => {
  const img = image(5, 5, (x) => [17 + x, 99, 200, 255]);
  const out = defringe(img);
  assert.equal(out.width, 5);
  assert.equal(out.height, 5);
  for (let i = 0; i < img.data.length; i++) {
    assert.equal(out.data[i], img.data[i], `byte ${i} unchanged`);
  }
});

test("a contaminated fringe pixel adjacent to a clean solid region is pulled toward the solid color", () => {
  // 3x1: [solid pure red] [fringe dark-red, alpha 128] [fully transparent].
  // Only solid source in range is the red pixel -> middle RGB pulled to pure red; alpha stays 128.
  const img = image(3, 1, (x) => {
    if (x === 0) return [255, 0, 0, 255]; // solid clean foreground
    if (x === 1) return [50, 0, 0, 128];  // contaminated fringe (dark from BG bleed)
    return [0, 0, 0, 0];                  // transparent
  });
  const out = defringe(img);
  assert.equal(out.width, 3);
  assert.equal(out.height, 1);
  // RGB decontaminated toward the only solid neighbor (pure red).
  assert.ok(out.data[4] > img.data[4], "fringe R pulled UP toward solid red");
  assert.equal(out.data[4], 255, "single solid source -> exactly its R");
  assert.equal(out.data[5], 0, "G");
  assert.equal(out.data[6], 0, "B");
  // Alpha untouched.
  assert.equal(out.data[7], 128, "fringe alpha unchanged");
  // The solid source pixel itself is untouched.
  assert.deepEqual([...out.data.slice(0, 4)], [255, 0, 0, 255]);
});

test("a fringe pixel with NO solid pixel in range keeps its original RGB (safe fallback)", () => {
  // A lone fringe pixel: no solid neighbor anywhere -> RGB carried through unchanged.
  const img = image(1, 1, () => [50, 60, 70, 128]);
  const out = defringe(img);
  assert.deepEqual([...out.data], [50, 60, 70, 128]);
});
