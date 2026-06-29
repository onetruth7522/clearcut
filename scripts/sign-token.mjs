// Mint a ClearCut pro-license token (Phase 2, Model B). Offline operator tool — run per sale:
//
//   node scripts/sign-token.mjs [--id <nonce>]
//
// Reads the Ed25519 private key from .keys/clearcut-ed25519.key (or $CLEARCUT_SIGNKEY, 64 hex
// chars) and prints a `<payloadB64u>.<sigB64u>` token to stdout — email it to the buyer, who
// pastes it into ClearCut to unlock Pro. The signature is over the ASCII bytes of the base64url
// payload STRING, the exact scheme src/license.ts verifies (no JSON-canonicalization drift).
// The optional --id stamps a nonce into the payload for your records; it does not affect verify.
import * as ed from "@noble/ed25519";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const keyPath = join(root, ".keys", "clearcut-ed25519.key");
const b64u = (b) => Buffer.from(b).toString("base64url");

function loadSecret() {
  const hex = (process.env.CLEARCUT_SIGNKEY ?? readFileSync(keyPath, "utf8")).trim();
  if (!/^[0-9a-fA-F]{64}$/.test(hex)) {
    throw new Error("private key must be 64 hex chars (32 bytes) — set $CLEARCUT_SIGNKEY or run gen-keypair.mjs");
  }
  return Uint8Array.from(Buffer.from(hex, "hex"));
}

const idIdx = process.argv.indexOf("--id");
const id = idIdx !== -1 ? process.argv[idIdx + 1] : undefined;

const payload = { v: 1, product: "clearcut-pro", iat: new Date().toISOString().slice(0, 10) };
if (id) payload.id = id;

const payloadB64u = b64u(Buffer.from(JSON.stringify(payload), "utf8"));
const sig = await ed.signAsync(new TextEncoder().encode(payloadB64u), loadSecret());
console.log(`${payloadB64u}.${b64u(sig)}`);
