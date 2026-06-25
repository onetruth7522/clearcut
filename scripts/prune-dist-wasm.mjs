// Vite emits a hashed copy of the asyncify ORT wasm into dist/assets/ because the
// onnxruntime-web glue contains a static `new URL("...asyncify.wasm", import.meta.url)`.
// At runtime we override `env.backends.onnx.wasm.wasmPaths` to dist/ort/, so ORT fetches
// the self-hosted copy from dist/ort/ and the dist/assets/ copy is NEVER requested
// (verified by removing it and confirming inference still completes). Prune it so first
// load and the published repo don't carry ~23 MB of dead weight. (CF-0012)
import { readdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const assets = join(dirname(fileURLToPath(import.meta.url)), "..", "dist", "assets");
let pruned = 0;
for (const f of readdirSync(assets)) {
  if (/\.wasm$/.test(f)) {
    rmSync(join(assets, f));
    pruned++;
    console.log(`prune-dist-wasm: removed dist/assets/${f}`);
  }
}
console.log(`prune-dist-wasm: ${pruned} dead wasm asset(s) pruned (self-hosted copies in dist/ort/ are the live ones)`);
