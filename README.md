# ClearCut

A **zero-backend, fully client-side** AI background remover. Drop an image, get a
full-resolution cut-out with a transparent background — **no watermark, no sign-up, and your
photos never leave your device.** Everything runs in your browser via WebGPU (with a WASM
fallback); there is no server.

## Why

Server-based removers (remove.bg et al.) upload your photo, watermark or downscale the free
output, and charge per image. ClearCut does the work locally at full resolution for free.

## How it works

- **Model:** [U²-Netp](https://huggingface.co/BritishWerewolf/U-2-Netp) (Apache-2.0, ~4.4 MB ONNX),
  run with [Transformers.js](https://github.com/huggingface/transformers.js) over ONNX Runtime Web.
- Inference runs in a **Web Worker** at 320×320; the resulting saliency mask is upscaled and
  composited onto the **original full-resolution pixels** on the main thread, so the downloaded
  PNG matches your source dimensions exactly.
- ONNX Runtime WASM is **self-hosted** in `dist/ort/` (not a CDN) for reproducible deploys.

## Develop

```bash
npm install
npm run dev        # vite dev server
npm test           # composite unit tests (the full-res spec)
npm run build      # type-check + emit static dist/
npm run preview    # serve the built dist/ locally (use this for the cold-drive)
```

## Deploy

`npm run build` emits a static `dist/` with **zero server runtime** — drop it on GitHub Pages or
Netlify free tier. (GitHub Pages cannot set COOP/COEP headers, so ClearCut uses single-threaded
WASM and works there without configuration.)

## Status

Phase 1 — free single-image core. Batch processing, bulk-ZIP export, a higher-quality model
option, and a tip jar are planned for Phase 2.

## Licensing

ClearCut is MIT-licensed (see `LICENSE`). Bundled third-party components and the model are
credited in [`THIRD-PARTY-NOTICES.md`](./public/THIRD-PARTY-NOTICES.md).
