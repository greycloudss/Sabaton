# Sabaton

Sabaton is a lightweight, offline cracking & keyspace toolkit written in C. It focuses on flexible keyspace/wordlist generation, modular cipher and hash engines, and running with minimal dependencies — suitable for air-gapped or resource-constrained environments. This README mirrors the concise style used for Gauntlet.

> Part of the **Armourer** series of projects.  
> **Work in Progress**



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
| Stream ciphers                             | ✓      | Done        |
| Stream ciphers, statistical tests          | ✓      | Done        |
| Knapsack cryptosystem                      | ✓      | Done        |
| RSA cipher (cryptosystem)                  | ✓      | Done        |
| Rabin and Blum–Goldwasser cryptosystems    | ✓      | Done        |
| ElGamal cryptosystem and digital signature | ✓      | Done        |
| Secret sharing                             | ✗      | —           |
| Secret sharing schemes                     | ✗      | —           |
| Elliptic-curve cryptography                | ✓      | Done        |
| Zero-knowledge proofs                      | ✓      | Done        |


---

## Quick Build

If there is a wish to use GPU acceleration one must have in the __.vscode__ folder the __c_cpp_properties.json__ file:
```json
{
    "configurations": [
        {
            "name": "Linux",
            "defines": [
                "USE_CUDA",
                "__CUDACC__"
            ],
            "intelliSenseMode": "linux-gcc-x64",
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "includePath": [
                "/usr/include",
                "/opt/cuda/include"
            ]
        }
    ],
    "version": 4
}
```
As you can see, it requires NVCC as well, thus an NVIDIA card is nescesary.  
The special compilation for GPU feature access (.sabaton -h)can be done with:
```bash
make CUDA=1
```
This way there can be access to the GPU acceleration but only for specific cyphers. If the user wants to compile with but wants to use the CPU version, they can do it as well without the -gpu version.  
If the user wants to compile normaly:
```bash
make
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

### ZKP helper (quadratic residue + discrete log transcript)

Deterministically produces Fiat–Shamir transcripts for the course tasks:

```bash
# variant numbers: c, p, g, y, p
./sabaton.exe -decypher -zkp -frag "63723,100003,2,10842,100003"
```

Output format matches the lecture examples:
```
1) P = [85262, 90441, 21019, 1234, 12345]
C = [90565, 29102, 85110, 73213, 64525]

2) c = 1906, r = 7169
```
Transcripts are deterministic for the same inputs; `seed:<n>` inside `-frag` is optional if you want to force a different transcript.

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
