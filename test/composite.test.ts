// Normative spec for the full-resolution composite (contract T1.5).
// These examples WIN over any prose in the contract or module.
import { test } from "node:test";
import assert from "node:assert/strict";
import { compositeAlpha, type RGBAImage, type MaskInput } from "../src/composite.ts";

function solidImage(w: number, h: number, r: number, g: number, b: number): RGBAImage {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return { data, width: w, height: h };
}

function uniformMask(w: number, h: number, value: number): MaskInput {
  return { data: new Uint8Array(w * h).fill(value), width: w, height: h };
}

test("output dimensions equal the ORIGINAL image, not 320x320 (full-res preserved)", () => {
  const original = solidImage(800, 600, 10, 20, 30);
  const mask = uniformMask(320, 320, 255);
  const out = compositeAlpha(original, mask);
  assert.equal(out.width, 800);
  assert.equal(out.height, 600);
  assert.equal(out.data.length, 800 * 600 * 4);
});

test("mask 0 -> alpha 0 (transparent background)", () => {
  const original = solidImage(800, 600, 10, 20, 30);
  const out = compositeAlpha(original, uniformMask(320, 320, 0));
  for (let i = 0; i < out.width * out.height; i++) {
    assert.equal(out.data[i * 4 + 3], 0);
  }
});

test("mask 255 -> alpha 255 AND original RGB preserved (foreground color unchanged)", () => {
  const original = solidImage(800, 600, 17, 99, 200);
  const out = compositeAlpha(original, uniformMask(320, 320, 255));
  for (let i = 0; i < out.width * out.height; i++) {
    assert.equal(out.data[i * 4], 17, "R preserved");
    assert.equal(out.data[i * 4 + 1], 99, "G preserved");
    assert.equal(out.data[i * 4 + 2], 200, "B preserved");
    assert.equal(out.data[i * 4 + 3], 255, "alpha opaque");
  }
});

test("identity case: 320x320 input + 320x320 mask maps 1:1 with no resize artifact", () => {
  const W = 320;
  const original = solidImage(W, W, 5, 5, 5);
  // checkerboard-ish mask: alternate 0/255 per pixel, must map exactly with no blending
  const md = new Uint8Array(W * W);
  for (let i = 0; i < md.length; i++) md[i] = i % 2 === 0 ? 255 : 0;
  const out = compositeAlpha(original, { data: md, width: W, height: W });
  for (let i = 0; i < W * W; i++) {
    assert.equal(out.data[i * 4 + 3], md[i], `alpha at ${i} maps exactly (no interpolation)`);
  }
});

test("partial mask: a known gradient upscales monotonically into alpha", () => {
  // 2x1 mask [0, 255] over a 4x1 image -> alpha should be non-decreasing left to right,
  // endpoints pinned to the mask values.
  const original = solidImage(4, 1, 0, 0, 0);
  const mask: MaskInput = { data: new Uint8Array([0, 255]), width: 2, height: 1 };
  const out = compositeAlpha(original, mask);
  const alphas = [out.data[3], out.data[7], out.data[11], out.data[15]];
  assert.equal(alphas[0], 0, "left endpoint = mask[0]");
  assert.equal(alphas[3], 255, "right endpoint = mask[1]");
  for (let i = 1; i < alphas.length; i++) {
    assert.ok(alphas[i] >= alphas[i - 1], "alpha non-decreasing across the gradient");
  }
});
