#include "../../../util/number.h"
#include "../../../util/string.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Simplified Blum–Goldwasser-style decrypt using provided seed x0 (no square-root step). */
extern "C" const char* blumGoldwasserCuda(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    static char* out = NULL;
    if (out) { free(out); out = NULL; }
    if (!encText || !*encText || !frag || !*frag) return "[bg cuda] missing input";

    /* frag: p|q|x0 */
    char* copy = strdup(frag);
    if (!copy) return "[bg cuda] OOM";
    char* tok = strtok(copy, "|");
    unsigned long long p = 0, q = 0, x = 0;
    if (tok) { p = strtoull(tok, NULL, 10); tok = strtok(NULL, "|"); }
    if (tok) { q = strtoull(tok, NULL, 10); tok = strtok(NULL, "|"); }
    if (tok) { x = strtoull(tok, NULL, 10); }
    free(copy);
    if (p < 3 || q < 3 || x == 0) return "[bg cuda] bad frag";
    unsigned __int128 n128 = (unsigned __int128)p * (unsigned __int128)q;
    if (n128 > (unsigned __int128)0xFFFFFFFFFFFFFFFFULL) return "[bg cuda] n too large";
    uint64_t n = (uint64_t)n128;

    int clen = 0;
    int* cbytes = parse_frag_array(encText, &clen);
    if (!cbytes || clen <= 0) { if (cbytes) free(cbytes); return "[bg cuda] bad ciphertext"; }

    char* plain = (char*)malloc((size_t)clen + 1);
    if (!plain) { free(cbytes); return "[bg cuda] OOM"; }

    uint64_t xi = x % n;
    for (int i = 0; i < clen; ++i) {
        xi = (uint64_t)((__uint128_t)xi * xi % n);
        uint8_t keystream = (uint8_t)(xi & 0xFFu);
        plain[i] = (char)((uint8_t)cbytes[i] ^ keystream);
    }
    plain[clen] = '\0';
    free(cbytes);
    out = plain;
    return out;
}

#endif
