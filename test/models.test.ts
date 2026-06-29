// Normative spec for the model descriptor registry + preprocess/extraction (PHASE-CONTRACT 2b, Amendment 1).
// These examples WIN over any prose in the contract or module.
import { test } from "node:test";
import assert from "node:assert/strict";
import { MODELS, maskFromOutput, preprocessNCHW, sigmoid } from "../src/models.ts";

// --- descriptors -----------------------------------------------------------

test("fast descriptor equals the Phase-1 U-2-Netp constants EXACTLY (free-path-unchanged anchor)", () => {
  assert.deepEqual(MODELS.fast, {
    key: "fast",
    runtime: "transformers",
    inputSize: 320,
    rescale: true,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    output: "saliency",
    id: "BritishWerewolf/U-2-Netp",
    dtype: "fp32",
    device: "auto",
    outputKey: "1959",
  });
});

test("hq descriptor is IS-Net general-use via raw ORT-Web on WASM (Amendment 1)", () => {
  const hq = MODELS.hq;
  assert.equal(hq.runtime, "ort-raw");
  assert.equal(hq.ep, "wasm");
  assert.equal(hq.inputSize, 1024);
  assert.equal(hq.rescale, false);
  assert.deepEqual(hq.mean, [128, 128, 128]);
  assert.deepEqual(hq.std, [256, 256, 256]);
  assert.equal(hq.output, "saliency"); // IS-Net "output" is already 0..1
  assert.equal(hq.inputName, "input");
  assert.equal(hq.outputName, "output");
  assert.match(hq.url ?? "", /isnet-general-use.*\.onnx$/);
});

// --- preprocessNCHW (the manual preprocess shared by both runtimes) --------

test("preprocessNCHW: IS-Net path (no rescale, (v-128)/256) — NCHW layout", () => {
  const chw = preprocessNCHW(new Uint8ClampedArray([200, 100, 50, 255]), 1, false, [128, 128, 128], [256, 256, 256]);
  assert.equal(chw.length, 3); // 3 channels * 1 plane
  assert.ok(Math.abs(chw[0] - (200 - 128) / 256) < 1e-6, "R");
  assert.ok(Math.abs(chw[1] - (100 - 128) / 256) < 1e-6, "G");
  assert.ok(Math.abs(chw[2] - (50 - 128) / 256) < 1e-6, "B");
});

test("preprocessNCHW: U-2-Netp path (rescale ÷255 then ImageNet) reproduces Phase-1 normalization", () => {
  const M = [0.485, 0.456, 0.406], S = [0.229, 0.224, 0.225];
  const chw = preprocessNCHW(new Uint8ClampedArray([255, 0, 128, 255]), 1, true, M as [number, number, number], S as [number, number, number]);
  assert.ok(Math.abs(chw[0] - (1 - M[0]) / S[0]) < 1e-6, "R");
  assert.ok(Math.abs(chw[1] - (0 - M[1]) / S[1]) < 1e-6, "G");
  assert.ok(Math.abs(chw[2] - (128 / 255 - M[2]) / S[2]) < 1e-6, "B");
});

// --- maskFromOutput: the §6 normative table -------------------------------

test("saliency extraction reproduces the existing Phase-1 U-2-Netp math exactly (also IS-Net 0..1)", () => {
  const cases: [number, number][] = [[1.0, 255], [0.0, 0], [0.5, 128]]; // 0.5 -> 127.5 -> round half-up 128
  for (const [v, expected] of cases) {
    assert.equal(maskFromOutput("saliency", [v])[0], expected, `saliency ${v} -> ${expected}`);
  }
});

test("logit-sigmoid extraction matches the normative examples (kept though currently unused)", () => {
  const cases: [number, number][] = [[0, 128], [10, 255], [-10, 0], [2, 225], [-2, 30]];
  for (const [logit, expected] of cases) {
    assert.equal(maskFromOutput("logit-sigmoid", [logit])[0], expected, `logit ${logit} -> ${expected}`);
  }
});

test("maskFromOutput returns a uint8 mask the same length as the input plane", () => {
  const out = maskFromOutput("saliency", new Float32Array(1024 * 1024));
  assert.ok(out instanceof Uint8Array);
  assert.equal(out.length, 1024 * 1024);
});

test("sigmoid is the standard logistic", () => {
  assert.equal(sigmoid(0), 0.5);
  assert.ok(Math.abs(sigmoid(2) - 0.8807970779778823) < 1e-12);
});
