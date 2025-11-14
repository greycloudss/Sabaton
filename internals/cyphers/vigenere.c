#include "vigenere.h"

static int decode_alphabet(const char* alphabetUtf8, uint32_t* alphabetOut, int capacity) {
    return utf8_to_u32(alphabetUtf8, alphabetOut, capacity);
}

static int decode_utf8_to_u32(const char* s, uint32_t* out, int capacity) {
    return utf8_to_u32(s, out, capacity);
}

static int starts_with_literal(const char* s, const char* prefix) {
    int n = slen(prefix);
    for (int i = 0; i < n; ++i) {
        if (!s[i] || s[i] != prefix[i]) return 0;
    }
    return 1;
}

static int is_lithuanian_alphabet(const uint32_t* alphabet, int m) {
    int hasY = 0;
    int hasDiacritic = 0;
    for (int i = 0; i < m; ++i) {
        uint32_t c = alphabet[i];
        if (c == 'Y') hasY = 1;
        if (c == 0x0104 || c == 0x010C || c == 0x0118 || c == 0x0116 || c == 0x012E || c == 0x0160 || c == 0x0172 || c == 0x016A || c == 0x017D) hasDiacritic = 1;
    }
    return hasY && hasDiacritic;
}

static void english_expected_26(double* out) {
    double v[26] = {
        0.082, 0.015, 0.028, 0.043, 0.127, 0.022, 0.020, 0.061, 0.070, 0.002,
        0.008, 0.040, 0.024, 0.067, 0.075, 0.019, 0.001, 0.060, 0.063, 0.091,
        0.028, 0.010, 0.023, 0.001, 0.020, 0.001
    };
    for (int i = 0; i < 26; ++i) {
        out[i] = v[i];
    }
}

static void expected_for_alphabet(const uint32_t* alphabet, int m, double* expected) {
    if (is_lithuanian_alphabet(alphabet, m)) {
        const char* canonicalUtf8 = "AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ";
        double f[32] = {
            0.10902180436087218, 0.007201440288057611, 0.014802960592118422, 0.005601120224044809,
            0.0036007201440288053, 0.02630526105221044, 0.056411282256451284, 0.001900380076015203,
            0.014702940588117623, 0.0034006801360272057, 0.017303460692138426, 0.002000400080016003,
            0.1369273854770954, 0.005701140228045609, 0.013702740548109622, 0.022304460892178433,
            0.04190838167633527, 0.03180636127225445, 0.03770754150830166, 0.05071014202840568,
            0.0638127625525105, 0.027505501100220042, 0.05721144228845769, 0.08011602320464092,
            0.009601920384076815, 0.0619123824764953, 0.04380876175235047, 0.014102820564112821,
            0.004000800160032006, 0.023904780956191238, 0.0034006801360272057, 0.007601520304060812
        };

        uint32_t canonical[64];

        int cn = decode_alphabet(canonicalUtf8, canonical, 64);

        for (int i = 0; i < m; ++i) {
            double val = 1.0 / (double)m;
            for (int j = 0; j < cn; ++j) {
                if (alphabet[i] == canonical[j]) {
                    val = f[j];
                    break;
                }
            }
            expected[i] = val;
        }

        double sum = 0.0;

        for (int i = 0; i < m; ++i) sum += expected[i];

        if (sum > 0.0) for (int i = 0; i < m; ++i) expected[i] /= sum;

        return;
    }
    double e26[26];
    english_expected_26(e26);

    for (int i = 0; i < m; ++i) {
        uint32_t c = alphabet[i];
        if (c >= 'A' && c <= 'Z') expected[i] = e26[c - 'A'];
        else expected[i] = 1.0 / (double)m;
    }

    double sum = 0.0;

    for (int i = 0; i < m; ++i) sum += expected[i];

    if (sum > 0.0) for (int i = 0; i < m; ++i) expected[i] /= sum;
}

static double chi_for_shift(const uint32_t* alphabet, int m, const uint32_t* column, int columnLength, int shift, const double* expected) {
    int* counts = (int*)malloc(sizeof(int) * m);

    for (int i = 0; i < m; ++i) counts[i] = 0;

    for (int i = 0; i < columnLength; ++i) {
        int cipherIndex = u32_index_of(alphabet, m, column[i]);
        if (cipherIndex < 0) continue;
        int plainIndex = mod(cipherIndex - shift, m);
        counts[plainIndex]++;
    }

    double chi = 0.0;

    for (int i = 0; i < m; ++i) {
        double expectedCount = expected[i] * (double)columnLength;
        if (expectedCount > 0.0) {
            double diff = (double)counts[i] - expectedCount;
            chi += diff * diff / expectedCount;
        }
    }

    free(counts);
    return chi;
}

static double total_chi_for_keylength(const uint32_t* alphabet, int m, const uint32_t* text, int textLength, int keyLength, const double* expected, const uint32_t* forcedPrefix, int forcedPrefixLength, int* shiftsOut) {
    uint32_t** columns = (uint32_t**)malloc(sizeof(uint32_t*) * keyLength);
    int* columnLengths = (int*)malloc(sizeof(int) * keyLength);

    for (int j = 0; j < keyLength; ++j) {
        columns[j] = (uint32_t*)malloc(sizeof(uint32_t) * textLength);
        columnLengths[j] = 0;
    }

    int position = 0;

    for (int i = 0; i < textLength; ++i) {
        if (u32_index_of(alphabet, m, text[i]) >= 0) {
            int j = position % keyLength;
            columns[j][columnLengths[j]++] = text[i];
            position++;
        }
    }
    double total = 0.0;
    for (int j = 0; j < keyLength; ++j) {
        int forced = -1;
        if (j < forcedPrefixLength) {
            int idx = u32_index_of(alphabet, m, forcedPrefix[j]);
            if (idx >= 0) forced = idx;
        }
        double bestChi = 1e100;
        int bestShift = 0;
        if (forced >= 0) {
            bestShift = forced;
            bestChi = chi_for_shift(alphabet, m, columns[j], columnLengths[j], forced, expected);
        } else {
            for (int s = 0; s < m; ++s) {
                double c = chi_for_shift(alphabet, m, columns[j], columnLengths[j], s, expected);
                if (c < bestChi) {
                    bestChi = c;
                    bestShift = s;
                }
            }
        }
        total += bestChi;
        if (shiftsOut) shiftsOut[j] = bestShift;
    }
    for (int j = 0; j < keyLength; ++j) free(columns[j]);

    free(columns);
    free(columnLengths);

    return total;
}

static void best_shifts_in_range(const uint32_t* alphabet, int m, const uint32_t* text, int textLength, int minLen, int maxLen, const double* expected, const uint32_t* forcedPrefix, int forcedPrefixLength, int* bestLenOut, int** bestShiftsOut) {
    if (minLen < 1) minLen = 1;
    if (maxLen < minLen) maxLen = minLen;
    if (maxLen > 64) maxLen = 64;
    if (forcedPrefixLength > minLen) if (minLen < forcedPrefixLength) minLen = forcedPrefixLength;

    double bestScore = 1e100;
    int chosenLen = minLen;
    int* chosenShifts = NULL;

    for (int L = minLen; L <= maxLen; ++L) {
        int* shifts = (int*)malloc(sizeof(int) * L);
        double score = total_chi_for_keylength(alphabet, m, text, textLength, L, expected, forcedPrefix, forcedPrefixLength, shifts);
        if (score < bestScore) {
            if (chosenShifts) free(chosenShifts);

            chosenShifts = shifts;
            bestScore = score;
            chosenLen = L;

        } else {
            free(shifts);
        }
    }

    *bestLenOut = chosenLen;
    *bestShiftsOut = chosenShifts;
}

static void parse_range_after_colon(const char* frag, int* Lmin, int* Lmax) {
    *Lmin = 1;
    *Lmax = 20;
    const char* p = frag;

    while (*p && *p != ':') ++p;

    if (*p == ':') {
        ++p;
        int a = 0;
        int b = 0;
        int ha = 0;
        int hb = 0;

        while (*p) {
            if (*p >= '0' && *p <= '9') {
                int digit = (*p - '0');
                a = a * 10 + digit;
                ha = 1;
                p++;
                continue;
            }
            if (*p == '-') {
                p++;
                while (*p >= '0' && *p <= '9') {
                    int digit = (*p - '0');
                    b = b * 10 + digit;
                    p++;
                }
                hb = 1;
                break;
            }
            break;
        }
        if (ha && hb) {
            *Lmin = a;
            *Lmax = b;
        } else if (ha) *Lmax = a;
    }

    if (*Lmin < 1) *Lmin = 1;
    if (*Lmax < *Lmin) *Lmax = *Lmin;
    if (*Lmax > 64) *Lmax = 64;
}


static int split_prefix_and_range(const char* frag, char* keybuf, int keycap, int* Lmin, int* Lmax) {
    const char* p = frag + 7;
    int w = 0;
    while (*p && *p != '|' && w + 4 < keycap) {
        uint32_t cp;
        int adv = utf8_decode_one(p, &cp);
        if (adv <= 0) break;
        for (int i = 0; i < adv; ++i) keybuf[w++] = p[i];
        p += adv;
    }

    keybuf[w] = '\0';
    *Lmin = 1;
    *Lmax = 20;

    if (*p == '|') {
        p++;

        int a = 0;
        int b = 0;
        int ha = 0;
        int hb = 0;

        while (*p) {
            if (*p >= '0' && *p <= '9') {
                int digit = (*p - '0');
                a = a * 10 + digit;
                ha = 1;
                p++;
                continue;
            }
            if (*p == '-') {
                p++;
                while (*p >= '0' && *p <= '9') {
                    int digit = (*p - '0');
                    b = b * 10 + digit;
                    p++;
                }
                hb = 1;
                break;
            }
            break;
        }

        if (ha && hb) {
            *Lmin = a;
            *Lmax = b;
        } else if (ha) {
            *Lmin = a;
            *Lmax = a;
        }
    }

    if (*Lmin < 1) *Lmin = 1;
    if (*Lmax < *Lmin) *Lmax = *Lmin;
    if (*Lmax > 64) *Lmax = 64;

    return w;
}

static void decrypt_vigenere_to_u32(const uint32_t* alphabet, int m, const uint32_t* text, int textLength, const uint32_t* key, int keyLength, uint32_t* out) {
    int pos = 0;

    for (int i = 0; i < textLength; ++i) {
        int ci = u32_index_of(alphabet, m, text[i]);

        if (ci < 0) out[i] = text[i];

        else {
            int ki = u32_index_of(alphabet, m, key[pos % keyLength]);
            int pi = mod(ci - ki, m);
            out[i] = alphabet[pi];
            pos++;
        }
    }
}

static void decrypt_vigenere_autokey_ciphertext_to_u32(const uint32_t* alphabet, int m, const uint32_t* text, int textLength, const uint32_t* key, int keyLength, uint32_t* out) {
    int pos = 0;
    int letters = 0;
    uint32_t* letterCipher = (uint32_t*)malloc(sizeof(uint32_t) * textLength);

    for (int i = 0; i < textLength; ++i) {
        int ci = u32_index_of(alphabet, m, text[i]);
        if (ci < 0) out[i] = text[i];
        else {
            int ki;
            if (pos < keyLength) ki = u32_index_of(alphabet, m, key[pos]);
            else {
                int cprev = u32_index_of(alphabet, m, letterCipher[pos - keyLength]);
                ki = cprev;
            }
            int pi = mod(ci - ki, m);
            out[i] = alphabet[pi];
            letterCipher[letters++] = text[i];
            pos++;
        }
    }

    free(letterCipher);
}

static const char* piece_vigenere_knownkey(const char* alphUtf8, const char* encUtf8, const char* keyUtf8) {
    static char out[32768];
    uint32_t A[256];
    uint32_t E[32768];
    uint32_t K[4096];
    uint32_t P[32768];

    int m = decode_alphabet(alphUtf8, A, 256);
    int n = decode_utf8_to_u32(encUtf8, E, 32768);
    int k = decode_utf8_to_u32(keyUtf8, K, 4096);

    if (m <= 1 || n <= 0 || k <= 0) return "";
    decrypt_vigenere_to_u32(A, m, E, n, K, k, P);

    int bytes = u32_to_utf8(P, n, out, (int)sizeof out);

    if (bytes <= 0) return "";

    return out;

}

static const char* piece_vigenere_crack(const char* alphUtf8, const char* encUtf8, int lenMin, int lenMax) {
    static char out[32768];

    uint32_t A[256];
    uint32_t E[32768];
    uint32_t P[32768];

    int m = decode_alphabet(alphUtf8, A, 256);
    int n = decode_utf8_to_u32(encUtf8, E, 32768);

    if (m <= 1 || n <= 0) return "";

    double* expected = (double*)malloc(sizeof(double) * m);

    expected_for_alphabet(A, m, expected);

    int bestLen;
    int* bestShifts;

    best_shifts_in_range(A, m, E, n, lenMin, lenMax, expected, NULL, 0, &bestLen, &bestShifts);

    uint32_t* key = (uint32_t*)malloc(sizeof(uint32_t) * bestLen);

    for (int i = 0; i < bestLen; ++i) key[i] = A[bestShifts[i]];

    decrypt_vigenere_to_u32(A, m, E, n, key, bestLen, P);

    free(key);
    free(bestShifts);
    free(expected);

    int bytes = u32_to_utf8(P, n, out, (int)sizeof out);

    if (bytes <= 0) return "";

    return out;
}

static const char* piece_vigenere_prefix(const char* alphUtf8, const char* encUtf8, const char* prefixUtf8, int lenMin, int lenMax) {
    static char out[32768];
    uint32_t A[256];
    uint32_t E[32768];
    uint32_t Pref[4096];
    uint32_t P[32768];

    int m = decode_alphabet(alphUtf8, A, 256);
    int n = decode_utf8_to_u32(encUtf8, E, 32768);
    int kp = decode_utf8_to_u32(prefixUtf8, Pref, 4096);

    if (m <= 1 || n <= 0 || kp <= 0) return "";

    double* expected = (double*)malloc(sizeof(double) * m);

    expected_for_alphabet(A, m, expected);

    int bestLen;
    int* bestShifts;

    best_shifts_in_range(A, m, E, n, lenMin, lenMax, expected, Pref, kp, &bestLen, &bestShifts);

    if (bestLen < kp) bestLen = kp;

    uint32_t* key = (uint32_t*)malloc(sizeof(uint32_t) * bestLen);

    for (int i = 0; i < bestLen; ++i) key[i] = A[bestShifts[i]];

    decrypt_vigenere_to_u32(A, m, E, n, key, bestLen, P);

    free(key);
    free(bestShifts);
    free(expected);

    int bytes = u32_to_utf8(P, n, out, (int)sizeof out);

    if (bytes <= 0) return "";

    return out;
}

static const char* piece_vigenere_autokey_ciphertext(const char* alphUtf8, const char* encUtf8, const char* keyUtf8) {
    static char out[32768];

    uint32_t A[256];
    uint32_t E[32768];
    uint32_t K[4096];
    uint32_t P[32768];

    int m = decode_alphabet(alphUtf8, A, 256);
    int n = decode_utf8_to_u32(encUtf8, E, 32768);
    int k = decode_utf8_to_u32(keyUtf8, K, 4096);

    if (m <= 1 || n <= 0 || k <= 0) return "";

    decrypt_vigenere_autokey_ciphertext_to_u32(A, m, E, n, K, k, P);

    int bytes = u32_to_utf8(P, n, out, (int)sizeof out);

    if (bytes <= 0) return "";

    return out;
}

static const char* brute_vigenere_candidates(const char* alphUtf8, const char* encUtf8) {
    static char fname[128];
    const char* base = "vigenereBrute-";

    int p = 0;

    while (base[p] && p < (int)sizeof(fname) - 1) {
        fname[p] = base[p];
        p++;
    }

    fname[p] = '\0';

    if (!append_time_txt(fname, (int)sizeof fname)) {
        const char* fb = "unknown.txt";
        int i = 0;
        while (fb[i] && p + i < (int)sizeof(fname) - 1) {
            fname[p + i] = fb[i];
            i++;
        }
        fname[p + i] = '\0';
    }

    FILE* fptr = fopen(fname, "wb");

    if (!fptr) return "";

    uint32_t A[256];
    uint32_t E[32768];
    uint32_t P[32768];

    int m = decode_alphabet(alphUtf8, A, 256);
    int n = decode_utf8_to_u32(encUtf8, E, 32768);

    if (m <= 1 || n <= 0) {
        fclose(fptr);
        return "";
    }

    double* expected = (double*)malloc(sizeof(double) * m);
    expected_for_alphabet(A, m, expected);

    for (int L = 1; L <= 20; ++L) {
        int* shifts = (int*)malloc(sizeof(int) * L);
        total_chi_for_keylength(A, m, E, n, L, expected, NULL, 0, shifts);
        uint32_t* key = (uint32_t*)malloc(sizeof(uint32_t) * L);

        for (int i = 0; i < L; ++i) key[i] = A[shifts[i]];

        decrypt_vigenere_to_u32(A, m, E, n, key, L, P);
        char lenbuf[16];
        int lenlen;
        i32_to_str(L, lenbuf, &lenlen);
        fwrite(lenbuf, 1, lenlen, fptr);
        fwrite(" ", 1, 1, fptr);

        char keybuf[4096];
        int keybytes = u32_to_utf8(key, L, keybuf, (int)sizeof keybuf);
        if (keybytes > 0) fwrite(keybuf, 1, keybytes, fptr);
        fwrite(" ", 1, 1, fptr);

        static char out[32768];
        int bytes = u32_to_utf8(P, n, out, (int)sizeof out);

        if (bytes > 0) fwrite(out, 1, bytes, fptr);

        fwrite("\n", 1, 1, fptr);
        free(key);
        free(shifts);
    }

    free(expected);
    fclose(fptr);
    return fname;
}

const char* vigenereEntry(const char* alphUtf8, const char* encUtf8, const char* frag) {
    if (!alphUtf8 || !encUtf8) return "";

    if (!frag || !frag[0]) {
        const char* r = brute_vigenere_candidates(alphUtf8, encUtf8);
        return (r && r[0]) ? r : "";
    }

    if (starts_with_literal(frag, "auto:")) {
        const char* key = frag + 5;
        return piece_vigenere_autokey_ciphertext(alphUtf8, encUtf8, key);
    }

    if (starts_with_literal(frag, "crack:")) {
        int Lmin, Lmax;
        parse_range_after_colon(frag, &Lmin, &Lmax);
        return piece_vigenere_crack(alphUtf8, encUtf8, Lmin, Lmax);
    }

    if (starts_with_literal(frag, "prefix:")) {
        char keybuf[512];
        int Lmin, Lmax;
        split_prefix_and_range(frag, keybuf, 512, &Lmin, &Lmax);
        return piece_vigenere_prefix(alphUtf8, encUtf8, keybuf, Lmin, Lmax);
    }
    return piece_vigenere_knownkey(alphUtf8, encUtf8, frag);
}
