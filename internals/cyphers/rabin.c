#include "rabin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "rabin_shared.h"

/* Lithuanian alphabet + space */
static const char* LIT_ALPH = "aąbcčdeęėfghiįyjklmnoprsštuųūvzž ";

/* ===============================
   Entry (CPU path; gated by -gpu in main)
   =============================== */
const char* rabinEntry(const char* alph, const char* encText, const char* frag) {
    static char* out = NULL;
    if (out) { free(out); out = NULL; }

    if ((!frag || !*frag) && (!encText || !*encText))
        return "[frag error]";

    const char* a = (alph && *alph) ? alph : LIT_ALPH;

    const char* c_s = encText && *encText ? encText : NULL;
    const char* p_s = NULL;
    const char* q_s = NULL;

    char* copy = frag ? strdup(frag) : NULL;
    if (copy) {
        char* save = NULL;
        char* t1 = strtok_r(copy, "|", &save);
        char* t2 = strtok_r(NULL, "|", &save);
        char* t3 = strtok_r(NULL, "|", &save);

        if (c_s) {
            /* encText holds ciphertext; frag can be p|q or c|p|q */
            if (t1 && t2 && !t3) {
                p_s = t1; q_s = t2;
            } else if (t1 && t2 && t3) {
                c_s = t1; p_s = t2; q_s = t3;
            }
        } else {
            /* everything inside frag */
            if (t1 && t2 && t3) {
                c_s = t1; p_s = t2; q_s = t3;
            }
        }
    }

    if (!c_s || !p_s || !q_s) {
        if (copy) free(copy);
        return "[frag error]";
    }

    BigInt c, p, q;
    biFromDec(&c, c_s);
    biFromDec(&p, p_s);
    biFromDec(&q, q_s);

    BigInt roots[4];
    int rcount = rabin_decrypt_roots(&c, &p, &q, roots);
    if (rcount <= 0) {
        if (copy) free(copy);
        return "[decode failed]";
    }

    char* best = NULL;
    for (int i = 0; i < rcount; ++i) {
        char* txt = rabin_number2text(&roots[i], a);
        if (!txt) continue;
        int ok = (strchr(txt, '?') == NULL);
        if (!best || ok) {
            if (best) free(best);
            best = txt;
            if (ok) break;
        } else {
            free(txt);
        }
    }

    if (!best) {
        if (copy) free(copy);
        return "[decode failed]";
    }

    char pbuf[256], qbuf[256], cbuf[256];
    biToDecString(&p, pbuf, sizeof(pbuf));
    biToDecString(&q, qbuf, sizeof(qbuf));
    biToDecString(&c, cbuf, sizeof(cbuf));

    size_t need = strlen(pbuf) + strlen(qbuf) + strlen(cbuf) + strlen(best) + 32;
    out = (char*)malloc(need);
    if (!out) {
        free(best);
        if (copy) free(copy);
        return "[oom]";
    }
    snprintf(out, need, "p=%s q=%s c=%s\n%s", pbuf, qbuf, cbuf, best);

    free(best);
    if (copy) free(copy);
    return out;
}
