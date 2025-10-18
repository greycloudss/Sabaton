# Sabaton

Sabaton is a lightweight, offline cracking & keyspace toolkit written in C. It focuses on flexible keyspace/wordlist generation, modular cipher and hash engines, and running with minimal dependencies — suitable for air-gapped or resource-constrained environments. This README mirrors the concise style used for Gauntlet.

> Part of the **Armourer** series of projects.  
> **Work in Progress**

---

## Features

* Multiple built-in hash implementations (SHA1, SHA256, CRC32, Murmur3, XXHash32, etc.)
* Cipher modules (Hill cipher, Affine/Caesar, and more)
* Fragment/key CSV parsing and custom alphabets (Unicode-aware)
* Keyspace / wordlist generation and brute-force helpers
* Minimal external dependencies: compiles with a standard C toolchain

---

## Status

WIP — core components exist but the project is actively being refined. Expect rough edges around: CLI behavior, padding conventions, error messages, and some cipher modules.

---

## For anyone coming from our cryptography course - the supported practice lecture topics:

| Translation (EN)                           | Status | Progress    |
| ------------------------------------------ | ------ | ----------- |
| Scytale cipher                             | ✓      | Done        |
| Transposition ciphers                      | ✓      | Done        |
| Fleissner grille cipher                    | ✓      | Done        |
| Delastelle/Bifid ciphers                   | ✓      | Done        |
| Analysis of substitution ciphers           | ✓      | Done        |
| Vigenère cipher analysis                   | ✓      | Done        |
| Enigma cipher                              | ✓      | Done        |
| Feistel cipher                             | ✓      | Done        |
| Block cipher modes (of operation)          | ✓      | Done        |
| AES variant                                | ✓      | Done        |
| Stream ciphers                             | ✗      | —           |
| Stream ciphers, statistical tests          | ✗      | —           |
| Knapsack cryptosystem                      | ✗      | —           |
| RSA cipher (cryptosystem)                  | ✗      | —           |
| Rabin and Blum–Goldwasser cryptosystems    | ✗      | —           |
| ElGamal cryptosystem and digital signature | ✗      | —           |
| Secret sharing                             | ✗      | —           |
| Secret sharing schemes                     | ✗      | —           |
| Elliptic-curve cryptography                | ✗      | —           |
| Zero-knowledge proofs                      | ✗      | —           |


---

## Quick Build

Adjust paths to match the repo layout. Example compile line used during development:

```bash
gcc .\main.c .\internals\hash.c .\internals\cyphers\affineCaesar.c .\internals\cyphers\enigma.c .\internals\cyphers\feistel.c .\internals\cyphers\hill.c .\internals\cyphers\scytale.c .\internals\cyphers\transposition.c .\internals\cyphers\vigenere.c .\internals\cyphers\block.c .\internals\hashes\crc32.c .\internals\hashes\murmur3.c .\internals\hashes\sha1.c .\internals\hashes\sha256.c .\internals\hashes\xxhash32.c .\internals\lithuanian.c .\internals\cyphers\aes.c .\internals\cyphers\fleissner.c .\internals\cyphers\bifid.c -lm
```
```bash
gcc main.c \
    internals/hash.c \
    internals/cyphers/affineCaesar.c \
    internals/hashes/crc32.c \
    internals/hashes/murmur3.c \
    internals/hashes/sha1.c \
    internals/hashes/sha256.c \
    internals/hashes/xxhash32.c \
    internals/cyphers/hill.c \
    internals/cyphers/vigenere.c \
    internals/cyphers/enigma.c \
    internals/cyphers/feistel.c \
    internals/lithuanian.c \
    -lm \
    -o sabaton
```


You can remove or add source files depending on which modules you want to include.

---

## Basic Usage

General invocation pattern:

```bash
./sabaton -decypher -hill -alph "<ALPHABET>" -frag "<fragment>" "<input>"
```

* `-decypher` — run in decryption mode (other modes exist depending on build)
* `-hill` — specify cipher module (replace with other cipher flags as available)
* `-alph` — alphabet string used for indexing (must include all characters in input)
* `-frag` — key fragment or CSV parameters (e.g., `17,6,4,9` for a 2×2 Hill key)
* final positional argument — ciphertext (or file depending on CLI)

Example:

```bash
./sabaton -decypher -hill -alph "AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ" -frag "17,6,4,9" "TBKKI HĄŪRH ..."
```

```bash
./sabaton -decypher -enigma -alph "ABCDEFGHIJKLMNOPQRSTUVWXYZ" \ 
  -frag "R1:5,3,2,0,17,10,8,24,20,11,1,12,9,22,16,6,25,4,18,21,7,13,15,23,19,14|R2:20,3,24,18,8,5,15,4,7,11,0,13,9,22,12,23,10,1,19,21,17,16,2,25,6,14|KEY:3,?" \
  "KAMMS PESAB ZDXXA VMYPZ ROJGF TMZGT TMNFZ GTQDL XQRPC DQQVR QFUQT TPOY"
  ```

Notes:

* Alphabets are processed as Unicode code points — make sure the provided `-alph` contains every codepoint used in the ciphertext.
* Hill cipher keys must be invertible mod alphabet size; non-invertible keys will not decrypt correctly.

---

## Design Notes

* The project intentionally keeps cryptographic primitives local (in `internals/`) so it can run without internet or large external libs.
* Unicode handling is performed by converting strings to code points and operating on indices, then encoding back to UTF-8.
* Padding/odd-length block handling differs across cipher modules — keep that in mind if results seem unexpected.

---

## Roadmap

* CLI help/usage output and improved argument validation
* Consistent, configurable padding for block ciphers
* Multithreading for faster keyspace searches
* Plugin system for adding/removing cipher & hash modules easily
* Better error messages when keys are invalid (e.g., non-invertible Hill matrices)
* Tests, examples, and sample datasets for common ciphers

---

## License

MIT — include `LICENSE` in the repo.

---
