#include "rabin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

//Something is wrong i think with BigInt lib
//./a.exe -decypher -rabin -alph "aąbcčdeęėfghiįyjklmnoprsštuųūvzž" -frag "942413682100603,942413682100651" "165660886409218178876532152962165660886409218178876532152962"

/* Parse "p,q" or "[p,q]" into BigInts */
static int parse_two_decimals(const char* frag, BigInt* p_out, BigInt* q_out) {
if (!frag) return 0;
char tmp[512];
size_t L = strlen(frag);
if (L >= sizeof(tmp)) return 0;

size_t s = 0, e = L;
if (frag[0] == '[') ++s;
if (frag[L - 1] == ']') --e;
size_t len = e - s;
memcpy(tmp, frag + s, len);
tmp[len] = '\0';

char* comma = strchr(tmp, ',');
if (!comma) return 0;
*comma = '\0';
char* pstr = tmp;
char* qstr = comma + 1;

while (*pstr == ' ' || *pstr == '\t') ++pstr;
while (*qstr == ' ' || *qstr == '\t') ++qstr;
char* endp = pstr + strlen(pstr) - 1;
while (endp >= pstr && (*endp == ' ' || *endp == '\t')) { *endp = '\0'; --endp; }
char* endq = qstr + strlen(qstr) - 1;
while (endq >= qstr && (*endq == ' ' || *endq == '\t')) { *endq = '\0'; --endq; }

if (!*pstr || !*qstr) return 0;

biFromDec(p_out, pstr);
biFromDec(q_out, qstr);
return 1;

}

static char* rabinDecodeDecimalToText(const BigInt* M, const char* alphUtf8) {
    if (!M || !alphUtf8) return NULL;

    /* produce decimal string for M */
    char* dec = biToAlphabet(M, "0123456789", 10);
    if (!dec) return NULL;

    /* ensure even length by prepending a '0' if needed */
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

    /* convert alphabet UTF-8 into code points array */
    uint32_t alph_cps[256];
    int base = utf8_to_u32(alphUtf8, alph_cps, 256);
    if (base <= 0) { free(dec); return NULL; }

    /* we'll decode into Unicode code points first */
    uint32_t msg_cps[8192];
    int count = 0;

    for (size_t i = 0; i < len; i += 2) {
        /* make sure these are digits */
        unsigned char c1 = (unsigned char)dec[i];
        unsigned char c2 = (unsigned char)dec[i + 1];
        if (!isdigit(c1) || !isdigit(c2)) {
            /* malformed digit pair -> place '?' and continue */
            msg_cps[count++] = '?';
            continue;
        }

        int d1 = c1 - '0';
        int d2 = c2 - '0';
        int code = d1 * 10 + d2; /* 0..99 */

        /* Python solver special-case: code == 33 means space */
        if (code == 33) {
            msg_cps[count++] = (uint32_t)' ';
        } else if (code >= 1 && code <= base) {
            /* map 01..base -> alph_cps[0..base-1] */
            msg_cps[count++] = alph_cps[code - 1];
        } else {
            /* out-of-range -> '?' */
            msg_cps[count++] = '?';
        }
    }

    free(dec);

    /* convert unicode cps to UTF-8 string */
    char* out = malloc((size_t)count * 4 + 1);
    if (!out) return NULL;
    u32_to_utf8(msg_cps, count, out, (int)(count * 4 + 1));
    return out;
}

/* Rabin decryption entry point */
const char* rabinEntry(const char* alph, const char* encText, const char* frag) {
if (!frag || !*frag) return strdup("[no frag provided]");
if (!encText || !*encText) return strdup("[no ciphertext provided]");

BigInt p, q;
if (!parse_two_decimals(frag, &p, &q))
    return strdup("[frag parse error: need p,q]");

if (biModU32(&p, 4) != 3 || biModU32(&q, 4) != 3) {
    biClear(&p); biClear(&q);
    return strdup("[p and q must both be 3 mod 4]");
}

BigInt n;
biMul(&n, &p, &q);

BigInt C;
biFromDec(&C, encText);

BigInt one, exp_p, exp_q, tmp;
biFromU32(&one, 1);

biAdd(&tmp, &p, &one);
biDivU32(&exp_p, &tmp, 4);
biAdd(&tmp, &q, &one);
biDivU32(&exp_q, &tmp, 4);
biClear(&tmp);

BigInt r_p, r_q;
biPowmod(&r_p, &C, &exp_p, &p);
biPowmod(&r_q, &C, &exp_q, &q);

BigInt inv_p;
if (!biModInv(&inv_p, &p, &q)) {
    biClear(&p); biClear(&q); biClear(&n);
    biClear(&C); biClear(&one);
    biClear(&exp_p); biClear(&exp_q);
    biClear(&r_p); biClear(&r_q);
    return strdup("[modinv failed]");
}

/* ---------- compute candidate roots (CRT) ---------- */
BigInt s1vals[2], s2vals[2], diff, h, ph, x;
biCopy(&s1vals[0], &r_p);
biCopy(&tmp, &p);
biSub(&tmp, &tmp, &r_p); 
biMod(&s1vals[1], &tmp, &p);

biCopy(&s2vals[0], &r_q);
biCopy(&tmp, &q);
biSub(&tmp, &tmp, &r_q);
biMod(&s2vals[1], &tmp, &q);

BigInt candidates[4];
for (int i = 0; i < 4; i++) biFromU32(&candidates[i], 0);

int cc = 0;
biFromU32(&diff, 0); biFromU32(&h, 0); biFromU32(&ph, 0); biFromU32(&x, 0);
for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
        biSub(&diff, &s2vals[j], &s1vals[i]);
        biMod(&diff, &diff, &q);
        biMulMod1(&h, &diff, &inv_p, &q);
        biMulMod1(&ph, &p, &h, &n);
        biAdd(&x, &s1vals[i], &ph);
        biMod(&x, &x, &n);

        biCopy(&candidates[cc++], &x);
    }
}

if (!alph || !*alph) {
    // no default alphabet allowed
    biClear(&p); biClear(&q); biClear(&n);
    biClear(&C); biClear(&one);
    biClear(&exp_p); biClear(&exp_q);
    biClear(&r_p); biClear(&r_q);
    biClear(&inv_p);
    biClear(&s1vals[0]); biClear(&s1vals[1]);
    biClear(&s2vals[0]); biClear(&s2vals[1]);
    biClear(&diff); biClear(&h); biClear(&ph); biClear(&x);
    for (int k = 0; k < cc; k++) biClear(&candidates[k]);
    return strdup("[no alphabet provided]");
}
const char* alphabet = alph;


static char* result = NULL;
if (result) { free(result); result = NULL; }

for (int k = 0; k < cc; k++) {
    char* txt = rabinDecodeDecimalToText(&candidates[k], alphabet);
    if (txt && txt[0]) { result = txt; break; }
    if (txt) free(txt);
}

/* ---------- cleanup ---------- */
biClear(&p); biClear(&q); biClear(&n);
biClear(&C); biClear(&one);
biClear(&exp_p); biClear(&exp_q);
biClear(&r_p); biClear(&r_q);
biClear(&inv_p);
biClear(&s1vals[0]); biClear(&s1vals[1]);
biClear(&s2vals[0]); biClear(&s2vals[1]);
biClear(&diff); biClear(&h); biClear(&ph); biClear(&x);
for (int k = 0; k < cc; k++) biClear(&candidates[k]);

if (result) return result;
return strdup("[decode failed]");

}
