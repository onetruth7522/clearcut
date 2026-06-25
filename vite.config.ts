import { defineConfig } from "vite";

// base: './' → relative asset paths so the built dist/ works under a GitHub Pages
// project subpath (e.g. user.github.io/clearcut/) as well as a domain root / local serve.
export default defineConfig({
  base: "./",
  worker: {
    // ESM worker output so the worker can `import` @huggingface/transformers.
    format: "es",
  },
  optimizeDeps: {
    // transformers.js pulls onnxruntime-web; let Vite pre-bundle it cleanly.
    exclude: ["@huggingface/transformers"],
  },
  build: {
    target: "es2022",
  },
});
