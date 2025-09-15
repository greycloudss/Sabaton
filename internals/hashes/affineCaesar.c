#include "affineCaesar.h"


void decryptWithKey_affineCaesar(const char* alph, const char* enc, int a, int b, FILE* fptr) {
    int alphLen = slen(alph);
    int aInverse;

    if (!modinv(a, alphLen, &aInverse)) return;

    for (int i = 0; enc[i]; ++i) {
        int abc = indexInAlphabet(alph, (unsigned char)enc[i]);
        char ch = (abc < 0) ? enc[i] : alph[mod(aInverse * (abc - b), alphLen)];

        fputc(ch, fptr);
    }
}


const char* pieceAffineCaesar(const char* alph, const char* encText, const char* knownFrag) {
    if (!alph || !encText || !knownFrag) return "";
    int m = slen(alph);
    int n = slen(encText);
    int k = slen(knownFrag);
    if (m < 2 || n < 2 || k < 2) return "";

    static char out[32768];
    if (n >= (int)sizeof(out)) return "";

    int limit = (k < n) ? k : n;

    for (int i = 0; i + 1 < limit; ++i) {
        int x1 = indexInAlphabet(alph, (unsigned char)knownFrag[i]);
        int x2 = indexInAlphabet(alph, (unsigned char)knownFrag[i + 1]);
        int y1 = indexInAlphabet(alph, (unsigned char)encText[i]);
        int y2 = indexInAlphabet(alph, (unsigned char)encText[i + 1]);
        if (x1 < 0 || x2 < 0 || y1 < 0 || y2 < 0) continue;

        int dx = mod(x2 - x1, m);
        if (dx == 0) continue;

        int dxinv;
        if (!modinv(dx, m, &dxinv)) continue;

        int a = mod((y2 - y1) * dxinv, m);
        int ainv;
        if (!modinv(a, m, &ainv)) continue;

        int b = mod(y1 - a * x1, m);

        for (int t = 0; t < n; ++t) {
            int yi = indexInAlphabet(alph, (unsigned char)encText[t]);
            out[t] = (yi < 0) ? encText[t] : alph[mod(ainv * (yi - b), m)];
        }
        out[n] = '\0';

        int ok = 1;
        for (int t = 0; t < k && t < n; ++t) {
            if (out[t] != knownFrag[t]) { ok = 0; break; }
        }
        if (ok) return out;
    }

    return "";
}

const char* bruteAffineCaesar(const char* alph, const char* encText) {
    static char fname[128];
    const char* base = "affineBruteCaesar-";
    int p = 0;
    while (base[p] && p < (int)sizeof(fname) - 1) { fname[p] = base[p]; ++p; }
    fname[p] = '\0';

    if (!append_time(fname, sizeof fname)) {
        const char* fb = "unknown.txt";
        int i = 0;
        while (fb[i] && p + i < (int)sizeof(fname) - 1) { fname[p + i] = fb[i]; ++i; }
        fname[p + i] = '\0';
    } else {
        const char* ext = ".txt";
        int e = 0, L = slen(fname);
        while (ext[e] && L + e < (int)sizeof(fname) - 1) { fname[L + e] = ext[e]; ++e; }
        fname[L + e] = '\0';
    }

    FILE* fptr = fopen(fname, "wb");
    if (!fptr) return "";

    int alphLen = slen(alph);

    for (int i = 1; i < alphLen; ++i) {
        int x, y;
        if (egcd(i, alphLen, &x, &y) != 1) continue;
        for (int j = 0; j < alphLen; ++j) {
            char abuf[16], bbuf[16];
            int al, bl;
            i32_to_str(i, abuf, &al);
            i32_to_str(j, bbuf, &bl);

            fwrite(abuf, 1, al, fptr);
            fwrite(" ", 1, 1, fptr);
            fwrite(bbuf, 1, bl, fptr);
            fwrite(" ", 1, 1, fptr);

            decryptWithKey_affineCaesar(alph, encText, i, j, fptr);
            fwrite("\n", 1, 1, fptr);
        }
    }

    fclose(fptr);
    largeWrite(fname);

    return '\0';
}

const char* affineCaesarEntry(const char* alph, const char* encText, const char* knownFrag, const char flag) {
    if (!alph || !encText)
        return "\0";

    if (knownFrag && flag == 0) { 
        const char* retVal = pieceaffineCaesar(alph, encText, knownFrag);
        return retVal && retVal[0] != '\0' ? retVal : "";

    } else {
        const char* retVal = bruteaffineCaesar(alph, encText);
        return retVal && retVal[0] != '\0' ? retVal : "";
    }
}