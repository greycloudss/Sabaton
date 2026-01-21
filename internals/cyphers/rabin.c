#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rabin.h"

//                                                                                           c                               p                  q
//./a.exe -decypher -rabin -alph "aąbcčdeęėfghiįyjklmnoprsštuųūvzž " -frag "21197264541260668663598848260720037099|9334134961424238127|9334134961424238143"

static char* number2text(const BigInt* M, const char* alphUtf8) {
    char* dec = biToAlphabet(M, "0123456789", 10);
    if (!dec) return NULL;

    size_t len = strlen(dec);
    if (len % 2 != 0) {
        char* tmp = malloc(len + 2);
        if (!tmp) { free(dec); return NULL; }
        tmp[0] = '0';
        memcpy(tmp + 1, dec, len + 1);
        free(dec);
        dec = tmp;
        len++;
    }

    uint32_t alph_cps[64];
    int base = utf8_to_u32(alphUtf8, alph_cps, 64);   
    if (base <= 0) { free(dec); return NULL; }

    uint32_t msg_cps[256];
    int count = 0;
    for (size_t i = 0; i < len && count < 256; i += 2) {
        int d1 = dec[i]   - '0';
        int d2 = dec[i+1] - '0';
        if (d1 < 0 || d1 > 9 || d2 < 0 || d2 > 9) continue;

        int code = d1 * 10 + d2;  
        if (code <= 0 || code > base) continue;

        msg_cps[count++] = alph_cps[code - 1];  
    }

    free(dec);

    char* out = malloc((size_t)count * 4 + 1);
    if (!out) return NULL;
    u32_to_utf8(msg_cps, count, out, (int)(count * 4 + 1));
    return out;
}

void __sqrt(BigInt *out, const BigInt *c, const BigInt *p) {
    BigInt one, exp, tmp, cmod;

    biMod(&cmod, c, p);

    biFromU32(&one, 1);
    biAdd(&tmp, p, &one);
    biDivU32(&exp, &tmp, 4);

    biPowMod(out, &cmod, &exp, p);
}


int decrypt_roots(
    const BigInt *c,
    const BigInt *p,
    const BigInt *q,
    BigInt roots[4]
) {
    BigInt mp, mq;
    __sqrt(&mp, c, p);
    __sqrt(&mq, c, q);

    BigInt yp, yq;
    if (!biModInv(&yp, p, q)) return 0;
    if (!biModInv(&yq, q, p)) return 0;

    BigInt n;
    biMul(&n, p, q);

    int idx = 0;
    for (int sp = 0; sp < 2; sp++) {
        BigInt rp;
        if (sp == 0) biCopy(&rp, &mp);
        else biSub(&rp, p, &mp);

        for (int sq = 0; sq < 2; sq++) {
            BigInt rq;
            if (sq == 0) biCopy(&rq, &mq);
            else biSub(&rq, q, &mq);

            BigInt t1, t2, sum, res;

            biMul(&t1, &rp, q);
            biMul(&t1, &t1, &yq);
            biMul(&t2, &rq, p);
            biMul(&t2, &t2, &yp);
            biAdd(&sum, &t1, &t2);
            biMod(&res, &sum, &n);
            biCopy(&roots[idx++], &res);
        }
    }
    return 4;
}

const char* rabinEntry(const char* alph,
                       const char* encText,
                       const char* frag)
{
    (void)alph;
    (void)encText;

    static char *out = NULL;
    if (out) {
        free(out);
        out = NULL;
    }

    if (!frag) return "ERROR: missing fragment";

    char *tmp = strdup(frag);
    if (!tmp) return "ERROR: out of memory";

    char *c_str = strtok(tmp, "|");
    char *p_str = strtok(NULL, "|");
    char *q_str = strtok(NULL, "|");

    if (!c_str || !p_str || !q_str) {
        free(tmp);
        return "ERROR: expected frag = c|p|q";
    }

    BigInt c, p, q;
    biFromDec(&c, c_str);
    biFromDec(&p, p_str);
    biFromDec(&q, q_str);

    BigInt roots[4];
    int cnt = decrypt_roots(&c, &p, &q, roots);

    if (cnt <= 0) {
        free(tmp);
        return "ERROR: Rabin decryption failed";
    }

    out = malloc(2048);
    if (!out) {
        free(tmp);
        return "ERROR: out of memory";
    }
    out[0] = '\0';

    for (int i = 0; i < cnt; i++) {
        char numbuf[512];
        biToDecString(&roots[i], numbuf, sizeof(numbuf));

        char *text = number2text(&roots[i], alph);
        if (!text) text = strdup("(decode error)");

        char line[1024];
        snprintf(line, sizeof(line),
                "Root %d = %s\nText   = %s\n\n",
                i + 1, numbuf, text);

        strcat(out, line);
        free(text);
    }


    free(tmp);
    return out;
}
