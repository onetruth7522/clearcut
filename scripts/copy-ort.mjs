// Vendor the ONNX Runtime Web wasm/glue into public/ort/ so Vite ships them in dist/ort/
// (self-hosted, not a CDN). These binaries are regenerable from node_modules, so they are
// gitignored and recreated by this script — run automatically before dev/build.
import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "onnxruntime-web", "dist");
const dst = join(root, "public", "ort");

mkdirSync(dst, { recursive: true });
// Ship only the two glue/wasm variants transformers.web.js can actually load:
//   - jsep      → imported via `onnxruntime-web`        (CPU / wasm device path)
//   - asyncify  → imported via `onnxruntime-web/webgpu` (WebGPU device path)
// The plain (`ort-wasm-simd-threaded.wasm`) and jspi builds are referenced by
// neither imported bundle entry, so vendoring them just bloats first-load (CF-0012).
const want = /^ort-wasm-simd-threaded\.(asyncify|jsep)\.(wasm|mjs)$/;
const files = readdirSync(src).filter((f) => want.test(f));
for (const f of files) copyFileSync(join(src, f), join(dst, f));
console.log(`copy-ort: ${files.length} ORT assets -> public/ort/`);
