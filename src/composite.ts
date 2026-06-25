// Pure full-resolution composite — the core differentiation.
//
// The model runs small (320x320) and returns a 320x320 saliency mask. This module takes that
// mask and the ORIGINAL full-resolution RGBA pixels, resizes the mask up to the original
// dimensions (bilinear), and writes it into the alpha channel. The output dimensions ALWAYS
// equal the input image's dimensions — never 320x320. That is what makes the saved PNG full-res.
//
// No DOM, no canvas — pure typed-array math, so it is unit-testable in node (test/composite.test.ts
// is the normative spec for this behavior).

export interface MaskInput {
  /** single-channel mask, 0..255, row-major width*height */
  data: Uint8Array;
  width: number;
  height: number;
}

export interface RGBAImage {
  data: Uint8ClampedArray; // RGBA, row-major
  width: number;
  height: number;
}

/**
 * Resize `mask` (maskW x maskH) up to `original`'s dimensions via bilinear interpolation and
 * apply it as the alpha channel of the original RGB. Foreground RGB is preserved unchanged.
 * Output dimensions == original dimensions.
 */
export function compositeAlpha(original: RGBAImage, mask: MaskInput): RGBAImage {
  const W = original.width;
  const H = original.height;
  const src = original.data;
  const mw = mask.width;
  const mh = mask.height;
  const md = mask.data;

  const out = new Uint8ClampedArray(W * H * 4);

  // Bilinear sample factors. When the output and mask dimensions match, the mapping is exact
  // (wx/wy become 0), so the identity case has no resize artifact.
  const sx = W > 1 ? (mw - 1) / (W - 1) : 0;
  const sy = H > 1 ? (mh - 1) / (H - 1) : 0;

  for (let y = 0; y < H; y++) {
    const my = y * sy;
    const my0 = Math.floor(my);
    const my1 = my0 + 1 < mh ? my0 + 1 : mh - 1;
    const wy = my - my0;

    for (let x = 0; x < W; x++) {
      const mx = x * sx;
      const mx0 = Math.floor(mx);
      const mx1 = mx0 + 1 < mw ? mx0 + 1 : mw - 1;
      const wx = mx - mx0;

      const a00 = md[my0 * mw + mx0];
      const a01 = md[my0 * mw + mx1];
      const a10 = md[my1 * mw + mx0];
      const a11 = md[my1 * mw + mx1];

      const top = a00 + (a01 - a00) * wx;
      const bot = a10 + (a11 - a10) * wx;
      const alpha = Math.round(top + (bot - top) * wy);

      const i = (y * W + x) * 4;
      out[i] = src[i]; // R — original color preserved
      out[i + 1] = src[i + 1]; // G
      out[i + 2] = src[i + 2]; // B
      out[i + 3] = alpha; // A — from the upscaled mask
    }
  }

  return { data: out, width: W, height: H };
}
