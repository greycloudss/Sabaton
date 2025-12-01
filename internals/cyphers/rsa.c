#include "rsa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 1
// ./sabaton.exe -decypher -rsa -alph "aąbcčdeęėfghiįyjklmnoprsštuųūvzž " \
-frag "[n,e,d]:[78320209297513663894769445385643, 71,76114006218710444812879770998471]" "54443358902833654672959014014776"

// 3
// ./sabaton.exe -decypher -rsa -alph "aąbcčdeęėfghiįyjklmnoprsštuųūvzž " -frag "mod:[n,e,c]:[3433579982949032254441327519214751514179009421,37,12211123975701803546598622421057741047118126]|[n,e,c]:[3433579982949032254441327519214751514179009421,117,878418226909325891356697307321765366345569673]" 

static char* rsaDecodeDecimalToText(const BigInt* M, const char* alphUtf8) {
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

static const char* rsaDeriveD(const char* n_str,
                              const char* e1_str,
                              const char* d1_str,
                              const char* e2_str)
{
    BigInt N, E1, D1, E2, K, one, Phi, D2;

    biFromDec(&N,  n_str);
    biFromDec(&E1, e1_str);
    biFromDec(&D1, d1_str);
    biFromDec(&E2, e2_str);

    biMul(&K, &E1, &D1);
    biFromU32(&one, 1);
    biSub(&K, &K, &one);

    int found_phi = 0;

    for (unsigned int k = 1; k <= 456; ++k) {
        if (biModU32(&K, k) != 0) {
            continue;
        }

        biDivU32(&Phi, &K, k); 

        if (biCmp(&Phi, &N) < 0) {
            found_phi = 1;
            break;              
        }

        biClear(&Phi);
    }

    if (!found_phi) {
        biClear(&N);
        biClear(&E1);
        biClear(&D1);
        biClear(&E2);
        biClear(&K);
        biClear(&one);
        return "RSA derive error: could not recover phi(n) from e1,d1";
    }

    if (!biModInv(&D2, &E2, &Phi)) {
        biClear(&N);
        biClear(&E1);
        biClear(&D1);
        biClear(&E2);
        biClear(&K);
        biClear(&one);
        biClear(&Phi);
        biClear(&D2);
        return "RSA derive error: e2 has no inverse mod phi(n)";
    }

    char* d2_str = biToAlphabet(&D2, "0123456789", 10);

    biClear(&N);
    biClear(&E1);
    biClear(&D1);
    biClear(&E2);
    biClear(&K);
    biClear(&one);
    biClear(&Phi);
    biClear(&D2);

    if (!d2_str) {
        return "RSA derive error: failed to convert d to decimal";
    }
    return d2_str;
}




static long long egcd_ll(long long a, long long b, long long* x, long long* y) {
    if (b == 0) {
        if (x) *x = 1;
        if (y) *y = 0;
        return a;
    }
    long long x1, y1;
    long long g = egcd_ll(b, a % b, &x1, &y1);
    if (x) *x = y1;
    if (y) *y = x1 - (a / b) * y1;
    return g;
}

static void biFromLL(BigInt* x, long long v) {
    if (v < 0) v = -v;
    biFromU32(x, (uint32_t)v);   
}


const char* rsaDecryption(const char* alph, const FragMap* vars, const char* encText){

    if (!vars || vars->count < 3) {
        return "RSA error: need at least 3 values (n,e,d) for simple decryption";
    }

    const char* n_str = vars->items[0].value;  
    const char* e_str = vars->items[1].value;  

    const char* c_str = encText;
    if (vars->count >= 3) {
        const char* n_str = vars->items[0].value;
        const char* d_str = vars->items[2].value;
        const char* c_str = encText;

        BigInt N, D, C, M;
        biFromDec(&N, n_str);
        biFromDec(&D, d_str);
        biFromDec(&C, c_str);

        biPowmod(&M, &C, &D, &N);

        int base = (int)strlen(alph);
        char* plaintext = rsaDecodeDecimalToText(&M, alph);

        biClear(&N);
        biClear(&D);
        biClear(&C);
        biClear(&M);

        return plaintext;
    }
}

const char* rsaModuloAttack(const char* alph,
                            const FragMap* blocks,
                            size_t block_count,
                            const char* encText)
{

    if (!blocks || block_count < 2) {
        return "RSA mod error: need at least 2 blocks (e,c) with common n";
    }

    const char* n_str = NULL;
    for (size_t i = 0; i < block_count; ++i) {
        n_str = fragmapGet(&blocks[i], "n");
        if (n_str && *n_str) break;
    }
    if (!n_str || !*n_str) {
        return "RSA mod error: missing modulus n in frag";
    }

    if (block_count < 2) {
        return "RSA mod error: need at least two (e,c) pairs";
    }

    const FragMap* b1 = &blocks[0];
    const FragMap* b2 = &blocks[1];

    const char* e1_str = fragmapGet(b1, "e");
    const char* c1_str = fragmapGet(b1, "c");
    const char* e2_str = fragmapGet(b2, "e");
    const char* c2_str = fragmapGet(b2, "c");

    if (!e1_str || !c1_str || !e2_str || !c2_str) {
        return "RSA mod error: each block must have e:... and c:...";
    }

    long long e1 = strtoll(e1_str, NULL, 10);
    long long e2 = strtoll(e2_str, NULL, 10);

    long long s, t;
    long long g = egcd_ll(e1, e2, &s, &t);

    if (g != 1 && g != -1) {
        return "RSA mod error: exponents are not coprime, common modulus attack fails";
    }

    if (g == -1) {
        s = -s;
        t = -t;
    }

    BigInt N, C1, C2;
    biFromDec(&N, n_str);
    biFromDec(&C1, c1_str);
    biFromDec(&C2, c2_str);

    BigInt term1, term2;

    if (s >= 0) {
        BigInt eS;
        biFromLL(&eS, s);
        biPowmod(&term1, &C1, &eS, &N);
        biClear(&eS);
    } else {
        BigInt inv1, eS;
        if (!biModInv(&inv1, &C1, &N)) {
            biClear(&N); biClear(&C1); biClear(&C2);
            return "RSA mod error: inverse of C1 does not exist mod n";
        }
        biFromLL(&eS, -s);
        biPowmod(&term1, &inv1, &eS, &N);
        biClear(&inv1);
        biClear(&eS);
    }

    if (t >= 0) {
        BigInt eT;
        biFromLL(&eT, t);
        biPowmod(&term2, &C2, &eT, &N);
        biClear(&eT);
    } else {
        BigInt inv2, eT;
        if (!biModInv(&inv2, &C2, &N)) {
            biClear(&N); biClear(&C1); biClear(&C2);
            biClear(&term1);
            return "RSA mod error: inverse of C2 does not exist mod n";
        }
        biFromLL(&eT, -t);
        biPowmod(&term2, &inv2, &eT, &N);
        biClear(&inv2);
        biClear(&eT);
    }

    BigInt M, tmp;
    biMulMod(&tmp, &term1, &term2, &N);
    biCopy(&M, &tmp);

    char* plaintext = rsaDecodeDecimalToText(&M, alph);

    biClear(&N);
    biClear(&C1);
    biClear(&C2);
    biClear(&term1);
    biClear(&term2);
    biClear(&M);
    biClear(&tmp);

    return plaintext ? plaintext : "RSA mod error: decode failed";
}


const char* rsaEntry(const char* alph, const char* encText, const char* frag)
{
    (void)encText; 

    int is_mod    = 0;
    int is_derive = 0;
    const char *spec = frag;

    if (strncmp(frag, "mod:", 4) == 0) {
        is_mod = 1;
        spec = frag + 4;
    } else if (strncmp(frag, "derive:", 7) == 0) {
        is_derive = 1;
        spec = frag + 7;
    }

    if (is_mod) {
        size_t block_count = 0;
        char **blocks = fragTokensSplit(spec, '|', &block_count);
        if (!blocks || block_count == 0) {
            fragTokensFree(blocks, block_count);
            return "RSA error: empty mod spec";
        }

        FragMap *maps = (FragMap*)calloc(block_count, sizeof(FragMap));
        if (!maps) {
            fragTokensFree(blocks, block_count);
            return "RSA error: OOM in modulo parsing";
        }

        for (size_t i = 0; i < block_count; ++i) {
            maps[i] = fragmapParseTupleSep(blocks[i], ':');
        }
        fragTokensFree(blocks, block_count);

        const char* result = rsaModuloAttack(alph, maps, block_count, encText);

        for (size_t i = 0; i < block_count; ++i) {
            fragmapFree(&maps[i]);
        }
        free(maps);
        return result;
    }
    else if (is_derive) {
        FragMap tup = fragmapParseTupleSep(spec, ':');
        const char* result = NULL;

        if (tup.count < 4) {
            result = "RSA error: derive expects [n,e1,d1,e2]";
        } else {
            const char* n_str  = tup.items[0].value;
            const char* e1_str = tup.items[1].value;
            const char* d1_str = tup.items[2].value;
            const char* e2_str = tup.items[3].value;

            if (!n_str || !e1_str || !d1_str || !e2_str) {
                result = "RSA error: derive tuple must be [n,e1,d1,e2]";
            } else {
                result = rsaDeriveD(n_str, e1_str, d1_str, e2_str);
            }
        }

        fragmapFree(&tup);
        return result;
    }
    else {
        FragMap tup = fragmapParseTupleSep(spec, ':');
        const char* result = rsaDecryption(alph, &tup, encText);
        fragmapFree(&tup);
        return result;
    }
}