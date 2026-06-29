// Generate the ClearCut pro-license Ed25519 keypair (Phase 2, Model B offline signer).
//
//   node scripts/gen-keypair.mjs [--force]
//
// The PRIVATE key is written to .keys/clearcut-ed25519.key (gitignored — NEVER committed or
// shipped); the PUBLIC key is printed for pasting into src/license.ts:PUBLIC_KEY. Run ONCE per
// key rotation. Refuses to overwrite an existing key unless --force: a clobbered private key
// strands every already-issued token, and there is no recovery (offline, no revocation).
import * as ed from "@noble/ed25519";
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const keyPath = join(root, ".keys", "clearcut-ed25519.key");
const hex = (b) => Buffer.from(b).toString("hex");

if (existsSync(keyPath) && !process.argv.includes("--force")) {
  console.error(
    `refusing to overwrite existing key at ${keyPath}\n` +
      `pass --force to rotate — this strands ALL already-issued tokens (no recovery).`,
  );
  process.exit(1);
}

const { secretKey, publicKey } = await ed.keygenAsync();
mkdirSync(dirname(keyPath), { recursive: true });
writeFileSync(keyPath, hex(secretKey) + "\n", { mode: 0o600 });

console.log(`private key -> ${keyPath}  (gitignored — never commit)`);
console.log(`public key  -> paste into src/license.ts:PUBLIC_KEY:`);
console.log(hex(publicKey));
