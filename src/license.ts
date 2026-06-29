// Pro entitlement — offline Ed25519 license verification (Phase 2, Model B).
//
// The DEPLOYED app makes ZERO licensing network calls: a pro "token" is a self-signed Ed25519
// assertion the user pastes, verified in-browser against the embedded public key below. Issuance
// is offline (scripts/sign-token.mjs, run by the operator); the private key never ships.
//
// Token string the user pastes:  <payloadB64u>.<sigB64u>
//   payloadB64u = base64url( UTF-8 JSON ), e.g. {"v":1,"product":"clearcut-pro","iat":"2026-06-29"}
//   sigB64u     = base64url( Ed25519 signature over the ASCII BYTES of the payloadB64u STRING )
// The signature is over the base64url string itself, NOT a re-serialized JSON object — this
// removes any JSON-canonicalization drift between the signer and this verifier (pinned spec;
// test/license.test.ts is the normative example set).
//
// Threat model: honor-ware, not DRM. This verifier ships to the client and is patchable; the goal
// is low-friction "enforceable-enough", not unbreakable enforcement. See PHASE-CONTRACT §Ratified.
import * as ed from "@noble/ed25519";

// Production Ed25519 public key (hex, 32 bytes). Its private half lives only in .keys/ (gitignored)
// and is used by the offline signer. Generated at the Phase 2 gate, 2026-06-29.
export const PUBLIC_KEY = "58dca5558df769bac2f192d214a30ef7e881664cbf76f5e14f0ee31e0452d70b";

const STORAGE_KEY = "clearcut.pro";

function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

// base64url -> bytes. `atob` is a global in browsers and modern Node (used by tests via tsx).
function b64uToBytes(s: string): Uint8Array {
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/");
  const pad = b64.length % 4 === 0 ? "" : "=".repeat(4 - (b64.length % 4));
  const bin = atob(b64 + pad);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const PUB = hexToBytes(PUBLIC_KEY);

/**
 * Verify a pasted token fully offline against `publicKey` (defaults to the embedded production key).
 * Resolves `true` only for a genuine signature over a well-formed `clearcut-pro` v1 payload.
 * NEVER throws — any malformed / tampered / wrong-product input resolves `false`.
 *
 * The `publicKey` parameter exists so the test suite can drive the normative example set with an
 * in-test keypair; production callers use the default. (test/license.test.ts is the spec.)
 */
export async function verifyToken(
  token: string | null | undefined,
  publicKey: Uint8Array = PUB,
): Promise<boolean> {
  try {
    if (typeof token !== "string") return false;
    const parts = token.trim().split(".");
    if (parts.length !== 2) return false;
    const [payloadB64u, sigB64u] = parts;
    if (!payloadB64u || !sigB64u) return false;
    const sig = b64uToBytes(sigB64u);
    const msg = new TextEncoder().encode(payloadB64u); // verify over the ASCII bytes of the string
    if (!(await ed.verifyAsync(sig, msg, publicKey))) return false;
    const payload = JSON.parse(new TextDecoder().decode(b64uToBytes(payloadB64u)));
    return payload != null && payload.v === 1 && payload.product === "clearcut-pro";
  } catch {
    return false;
  }
}

/** Verify a pasted token against the embedded production key. Thin binding over {@link verifyToken}. */
export function verifyProToken(token: string): Promise<boolean> {
  return verifyToken(token, PUB);
}

/**
 * Re-verify a stored entitlement string. Entitlement is the token's *signature*, not a boolean —
 * a hand-set `localStorage.clearcut.pro = "true"` is not a valid token and resolves `false`.
 * (`publicKey` is injectable for tests, same as {@link verifyToken}.)
 */
export function verifyStoredEntitlement(
  stored: string | null,
  publicKey: Uint8Array = PUB,
): Promise<boolean> {
  return verifyToken(stored, publicKey);
}

// --- localStorage glue (DOM-side; cold-drive-covered, not unit-tested — no storage in plain node) -
export function loadPro(): string | null {
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}
export function savePro(token: string): void {
  try { localStorage.setItem(STORAGE_KEY, token); } catch { /* private mode / storage disabled */ }
}
export function clearPro(): void {
  try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
}
