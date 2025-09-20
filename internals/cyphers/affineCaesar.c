#include "affineCaesar.h"
#include <stdint.h>

static int umin(int a, int b) { return a < b ? a : b; }

void decryptWithKey_affineCaesar(const char* alph, const char* enc, int a, int b, FILE* fptr) {
    if (!alph || !enc || !fptr) return;
    uint32_t alphcp[512];
    int m = utf8_to_u32(alph, alphcp, 512);
    if (m <= 1) return;
    int ainv;
    if (!modinv(a, m, &ainv)) return;

    const char* p = enc;
    while (*p) {
        uint32_t cp; int adv = utf8_decode_one(p, &cp);
        if (adv <= 0) break;
        int yi = u32_index_of(alphcp, m, cp);
        if (yi < 0) {
            for (int i = 0; i < adv; ++i) fputc(p[i], fptr);
        } else {
            int idx = mod(ainv * (yi - b), m);
            char buf[4]; int len = utf8_encode_one(alphcp[idx], buf);
            for (int i = 0; i < len; ++i) fputc(buf[i], fptr);
        }
        p += adv;
    }
}

const char* pieceAffineCaesar(const char* alph, const char* encText, const char* knownFrag) {
    if (!alph || !encText || !knownFrag) return "";
    uint32_t A[512], E[32768], K[4096], outcp[32768];
    int m = utf8_to_u32(alph, A, 512);
    int n = utf8_to_u32(encText, E, 32768);
    int k = utf8_to_u32(knownFrag, K, 4096);
    if (m < 2 || n < 2 || k < 2) return "";

    static char out[32768];
    int limit = umin(k, n);

    for (int i = 0; i + 1 < limit; ++i) {
        int x1 = u32_index_of(A, m, K[i]);
        int x2 = u32_index_of(A, m, K[i + 1]);
        int y1 = u32_index_of(A, m, E[i]);
        int y2 = u32_index_of(A, m, E[i + 1]);
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
            int yi = u32_index_of(A, m, E[t]);
            outcp[t] = (yi < 0) ? E[t] : A[mod(ainv * (yi - b), m)];
        }
        int bytes = u32_to_utf8(outcp, n, out, (int)sizeof out);

        int ok = 1;
        for (int t = 0; t < k && t < n; ++t) {
            if (outcp[t] != K[t]) { ok = 0; break; }
        }
        if (ok && bytes > 0) return out;
    }

    return "";
}

const char* bruteAffineCaesar(const char* alph, const char* encText) {
    if (!alph || !encText) return "";

    uint32_t A[512];
    int m = utf8_to_u32(alph, A, 512);
    if (m <= 1) return "";

    static char fname[128];
    const char* base = "affineBruteCaesar-";
    int p = 0;
    while (base[p] && p < (int)sizeof(fname) - 1) { fname[p] = base[p]; ++p; }
    fname[p] = '\0';
    if (!append_time_txt(fname, (int)sizeof fname)) {
        const char* fb = "unknown.txt";
        int i = 0;
        while (fb[i] && p + i < (int)sizeof(fname) - 1) { fname[p + i] = fb[i]; ++i; }
        fname[p + i] = '\0';
    }

    FILE* fptr = fopen(fname, "wb");
    if (!fptr) return "";

    for (int a = 1; a < m; ++a) {
        int x, y;
        if (egcd(a, m, &x, &y) != 1) continue;
        for (int b = 0; b < m; ++b) {
            char abuf[16], bbuf[16];
            int al, bl;
            i32_to_str(a, abuf, &al);
            i32_to_str(b, bbuf, &bl);
            fwrite(abuf, 1, al, fptr);
            fwrite(" ", 1, 1, fptr);
            fwrite(bbuf, 1, bl, fptr);
            fwrite(" ", 1, 1, fptr);
            decryptWithKey_affineCaesar(alph, encText, a, b, fptr);
            fwrite("\n", 1, 1, fptr);
        }
    }

    fclose(fptr);
    return fname;
}

const char* affineCaesarEntry(const char* alph, const char* encText, const char* knownFrag) {
    if (!alph || !encText) return "";
    if (knownFrag) {
        const char* r = pieceAffineCaesar(alph, encText, knownFrag);
        return (r && r[0]) ? r : "";
    } else {
        const char* r = bruteAffineCaesar(alph, encText);
        return (r && r[0]) ? r : "";
    }
}
/*
ŽODISYRASTIPRESNISUŽKARDĄJEIJISPASAKYTASTINKAMUMETU
17 4 TIKRASISIŠŠŪKISYRANEIŠŠIFRAVIMASOSUPRATIMASKĄTAIREIŠKIA
5 8 JEINORIPASIEKTIVIRŠŪNĘTURĖSIPRADĖTIŽINGSNĮNUOAPAČIOS
*/