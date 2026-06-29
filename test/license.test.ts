// Normative spec for offline pro-token verification (contract SC1 + SC2).
// These examples WIN over any prose in the contract or module.
//
// Tokens are minted in-test with an ephemeral keypair (the production private key never exists in
// the repo). The sign helper mirrors the pinned scheme exactly: sign the ASCII bytes of the
// base64url payload STRING — the same bytes scripts/sign-token.mjs signs and src/license.ts verifies.
import { test } from "node:test";
import assert from "node:assert/strict";
import * as ed from "@noble/ed25519";
import { verifyToken, verifyStoredEntitlement } from "../src/license.ts";

const b64u = (bytes: Uint8Array): string => Buffer.from(bytes).toString("base64url");
const utf8 = (s: string): Uint8Array => new TextEncoder().encode(s);

async function mint(
  secretKey: Uint8Array,
  payload: Record<string, unknown>,
): Promise<{ token: string; payloadB64u: string }> {
  const payloadB64u = b64u(utf8(JSON.stringify(payload)));
  const sig = await ed.signAsync(utf8(payloadB64u), secretKey);
  return { token: `${payloadB64u}.${b64u(sig)}`, payloadB64u };
}

const PRO = { v: 1, product: "clearcut-pro", iat: "2026-06-29" };

// --- SC1: verifyToken normative example set ---------------------------------------------------

test("SC1.1 — valid sig + clearcut-pro + v1 -> true", async () => {
  const { secretKey, publicKey } = await ed.keygenAsync();
  const { token } = await mint(secretKey, PRO);
  assert.equal(await verifyToken(token, publicKey), true);
});

test("SC1.2 — valid sig but wrong product -> false", async () => {
  const { secretKey, publicKey } = await ed.keygenAsync();
  const { token } = await mint(secretKey, { v: 1, product: "other", iat: "2026-06-29" });
  assert.equal(await verifyToken(token, publicKey), false);
});

test("SC1.3 — tampered payload byte -> false", async () => {
  const { secretKey, publicKey } = await ed.keygenAsync();
  const { token, payloadB64u } = await mint(secretKey, PRO);
  const sigPart = token.split(".")[1];
  const tampered = payloadB64u.slice(0, -1) + (payloadB64u.endsWith("A") ? "B" : "A");
  assert.equal(await verifyToken(`${tampered}.${sigPart}`, publicKey), false);
});

test("SC1.4 — tampered signature -> false", async () => {
  const { secretKey, publicKey } = await ed.keygenAsync();
  const { token, payloadB64u } = await mint(secretKey, PRO);
  const sig = Buffer.from(token.split(".")[1], "base64url");
  sig[0] ^= 0xff;
  assert.equal(await verifyToken(`${payloadB64u}.${sig.toString("base64url")}`, publicKey), false);
});

test("SC1.5 — malformed input never throws and is false", async () => {
  const { publicKey } = await ed.keygenAsync();
  for (const bad of ["", "nodot", "a.b.c", "....", "!!!.@@@", "  ", "x."]) {
    assert.equal(await verifyToken(bad, publicKey), false, `"${bad}" -> false`);
  }
  // also the production-key path tolerates junk without throwing
  assert.equal(await verifyToken("garbage"), false);
  assert.equal(await verifyToken(null), false);
  assert.equal(await verifyToken(undefined), false);
});

// --- SC2: verifyStoredEntitlement ------------------------------------------------------------

test("SC2 — a validly-signed stored token -> true; bare flag / junk / null -> false", async () => {
  const { secretKey, publicKey } = await ed.keygenAsync();
  const { token } = await mint(secretKey, PRO);
  assert.equal(await verifyStoredEntitlement(token, publicKey), true);
  assert.equal(await verifyStoredEntitlement("true", publicKey), false); // hand-set flag must not unlock
  assert.equal(await verifyStoredEntitlement("not-a-token", publicKey), false);
  assert.equal(await verifyStoredEntitlement(null, publicKey), false);
});
