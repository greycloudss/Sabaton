# CUDA Acceleration Guide

This project ships several CUDA-enhanced paths. Each one either brute-forces a constrained keyspace or offloads decrypt primitives to the GPU. Below is a concise map of what exists, how to invoke it, and what each mode does.

## Build
- Enable CUDA: `make CUDA=1`
- GPU dispatcher is selected with `-gpu` on the CLI.

## Modules

### AES variant (`-aes -gpu`)
- **Mode**: brute-force small 4-byte keyspace (`max` ≤ 31) for the custom AES-V rounds.
- **Params (frag)**: `brute|max:<n>|p:<p>|a:<a>|b:<b>|T:x,y,z,w|R:<rounds>`
- **Input**: ciphertext as int array string.
- **Output**: plaintext + recovered key.

### Stream cipher (`-stream -gpu`)
- **Mode**: brute-force LFSR taps/state for `lfsr:8`.
- **Params**: `8;brute` (fixed demo).
- **Input**: ciphertext ints.
- **Output**: plaintext + taps/state metadata.

### Knapsack / Merkle (`-merkle -gpu`)
- **Mode**: brute-force subset mask for superincreasing sequence (≤24 bits).
- **Params**: `key:[w1,...,wn]`
- **Input**: ciphertext ints.
- **Output**: recovered plaintext + key vector.

### RSA / Rabin (`-rsa -gpu`, `-rabin -gpu`)
- **RSA brute**: bounded Pollard Rho on GPU (then trial division fallback) to factor small `n`, then decrypt.
- **RSA modexp**: direct decrypt when `[n,e,d]` provided.
- **Rabin brute**: small `n` factoring + decrypt (same Pollard Rho path).
- **Blum–Goldwasser (BG)**: `-rabin -frag "bg:p|q|seed"` for direct decrypt; `bg:p|q|brute:<maxSeed>` to brute seeds over `[1..maxSeed]`.
- **Input**: ciphertext ints.
- **Output**: plaintext; for RSA brute, factors are used internally.

### Elliptic MV decrypt (`-ellipticCurve -gpu`)
- **Direct decrypt**: provide private scalar `r`.
- **Brute**: provide `maxr:<limit>` to search `r` in `[1..maxr]` with alphabet filtering.
- **Params**: `mode:mv|q:<prime>|a:<a>|b:<b>|P:[x,y]|n:<n>|r:<priv>` or `maxr:<limit>`.
- **Input**: blocks `[Rx,Ry,c1,c2,...]`.
- **Output**: plaintext bytes (pairs per block).

### ElGamal decrypt (`-elgamal -gpu`)
- **Direct decrypt**: provide private `a`.
- **Brute**: provide `beta:<g^a mod p>` and `max:<limit>` to search `a` in `[1..max]` with alphabet filtering.
- **Params**: `p:<prime>|g:<gen>|a:<priv>` or `beta:<pub>|max:<limit>`.
- **Input**: `[c1,c2,...]` pairs.
- **Output**: plaintext bytes.

### Shamir / Asmuth secret sharing (`-shamir -gpu`, `-asmuth -gpu`)
- **Mode**: reconstruct secret (not a brute force). If a plaintext is passed as positional argument, it is returned directly; otherwise digits are decoded via the alphabet.
- **Params**: Shamir `x1,x2,x3|s1,s2,s3|p`; Asmuth `shares|moduli|p_mod`.
- **Input**: optional plaintext override; otherwise numeric shares only.
- **Output**: reconstructed plaintext/secret.

### Feistel (`-feistel -gpu`)
- **Mode**: brute-force Feistel keys for 1–4 rounds with selectable round function.
- **Input**: ciphertext ints (even length).
- **Output**: candidate plaintext and keys (keys also logged to `feistel-keys-*.txt`).

## Sample script
- `cuda.sh` generates demo ciphertexts and runs all GPU paths, writing results to `gpu_*.txt` (the plaintext is the course’s “ONE … THIRTY” sentence). Adapt as needed for your inputs.

## Notes & Limits
- Brute-force searches are explicitly bounded by `max`/`maxr` to keep GPU work finite.
- RSA/Rabin “brute” and BG seed brute are only viable for small parameters.
- ElGamal/Elliptic brute modes rely on alphabet filtering to prune candidates; use realistic limits.
- Signature brute-force is not implemented; only decrypt/mask recovery paths are GPU-accelerated.

## Why Some Things Aren’t Implemented (or are Bounded)
- **Signatures (ElGamal/EC)**: Brute-forcing a signing key or nonce is a discrete-log search. Without tiny bounds it’s astronomically large; a GPU won’t make it practical.
- **Secret sharing brute**: Shamir/Asmuth reconstruction is deterministic; “bruting” shares or the secret without constraints is unbounded and meaningless cryptanalytically.
- **Factoring (RSA/Rabin/BG)**: Generic factoring is hard; the current brute modes only do small-factor scans. Realistic key sizes would require advanced algorithms, not simple GPU trial division.
- **Blum–Goldwasser seed brute**: Only seed search is provided. Bruting `p`/`q` collapses back to the factoring problem above.
