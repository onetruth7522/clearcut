// Normative spec for the model descriptor registry + mask extraction (PHASE-CONTRACT 2b §6).
// These examples WIN over any prose in the contract or module.
import { test } from "node:test";
import assert from "node:assert/strict";
import { MODELS, maskFromOutput, sigmoid } from "../src/models.ts";

// --- descriptors -----------------------------------------------------------

test("fast descriptor equals the Phase-1 U-2-Netp constants EXACTLY (free-path-unchanged anchor)", () => {
  assert.deepEqual(MODELS.fast, {
    key: "fast",
    id: "BritishWerewolf/U-2-Netp",
    dtype: "fp32",
    device: "auto",
    inputSize: 320,
    preprocess: "manual-squash",
    output: "saliency",
    outputKey: "1959",
  });
});

test("hq descriptor carries the BiRefNet_lite fp16 / webgpu / 1024 / auto-processor / logit-sigmoid shape", () => {
  assert.equal(MODELS.hq.id, "onnx-community/BiRefNet_lite-ONNX"); // pinned at gate (§5)
  assert.equal(MODELS.hq.dtype, "fp16");
  assert.equal(MODELS.hq.device, "webgpu");
  assert.equal(MODELS.hq.inputSize, 1024);
  assert.equal(MODELS.hq.preprocess, "auto-processor");
  assert.equal(MODELS.hq.output, "logit-sigmoid");
});

// --- maskFromOutput: the §6 normative table -------------------------------

test("logit-sigmoid extraction matches the normative examples", () => {
  // [logit input, expected uint8 out]
  const cases: [number, number][] = [
    [0, 128],   // sigmoid 0.5  -> 127.5 -> round half-up 128
    [10, 255],  // ~0.99995     -> ~254.988 -> 255
    [-10, 0],   // ~4.54e-5     -> ~0.0116 -> 0
    [2, 225],   // ~0.880797    -> ~224.60 -> 225
    [-2, 30],   // ~0.119203    -> ~30.40 -> 30
  ];
  for (const [logit, expected] of cases) {
    const out = maskFromOutput("logit-sigmoid", [logit]);
    assert.equal(out[0], expected, `logit ${logit} -> ${expected}`);
  }
});

test("saliency extraction reproduces the existing Phase-1 U-2-Netp math exactly", () => {
  const cases: [number, number][] = [
    [1.0, 255],
    [0.0, 0],
    [0.5, 128], // 127.5 -> round half-up 128
  ];
  for (const [v, expected] of cases) {
    const out = maskFromOutput("saliency", [v]);
    assert.equal(out[0], expected, `saliency ${v} -> ${expected}`);
  }
});

test("maskFromOutput returns a uint8 mask the same length as the input plane", () => {
  const out = maskFromOutput("logit-sigmoid", new Float32Array(1024 * 1024));
  assert.ok(out instanceof Uint8Array);
  assert.equal(out.length, 1024 * 1024);
});

test("sigmoid is the standard logistic", () => {
  assert.equal(sigmoid(0), 0.5);
  assert.ok(Math.abs(sigmoid(2) - 0.8807970779778823) < 1e-12);
  assert.ok(sigmoid(40) <= 1 && sigmoid(-40) >= 0);
});
