# Third-Party Notices

ClearCut runs entirely in your browser and is built on the following open-source components.
This file is part of the served application (shipped in `dist/`) to satisfy the attribution
obligations of the licenses below.

---

## U²-Net (background-removal model)

- **Model weights:** [`BritishWerewolf/U-2-Netp`](https://huggingface.co/BritishWerewolf/U-2-Netp)
  (an ONNX export of U²-Netp for Transformers.js).
- **Original work:** U²-Net — *"U²-Net: Going Deeper with Nested U-Structure for Salient Object
  Detection"*, Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R. Zaiane,
  Martin Jagersand. <https://github.com/xuebinqin/U-2-Net>
- **License:** Apache License 2.0.

```
Copyright (c) Xuebin Qin and the U-2-Net authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 🤗 Transformers.js

- **Project:** [`@huggingface/transformers`](https://github.com/huggingface/transformers.js) (v4.2.0).
- **License:** Apache License 2.0. Copyright (c) Hugging Face.

## ONNX Runtime Web

- **Project:** [`onnxruntime-web`](https://github.com/microsoft/onnxruntime) (bundled with Transformers.js).
- **License:** MIT License. Copyright (c) Microsoft Corporation.

## @noble/ed25519

- **Project:** [`@noble/ed25519`](https://github.com/paulmillr/noble-ed25519) (Ed25519 signatures;
  used to verify Pro license tokens entirely in your browser — no license server).
- **License:** MIT License. Copyright (c) Paul Miller.

## client-zip

- **Project:** [`client-zip`](https://github.com/Touffy/client-zip) (in-browser ZIP streaming for
  the Pro bulk-export download).
- **License:** MIT License. Copyright (c) David Junger.
