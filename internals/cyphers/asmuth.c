#include "asmuth.h"

void biToDecString(const BigInt* x, char* out, size_t out_len);

//./a.exe -decypher -asmuth -alph "AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ" -frag "69638025634,31469233709,1818908519|128658463541,213565974133,258735195703|95455728217"

static void parseCSVll(const char* s, long long* out, int* count) {
    *count = 0;
    if (!s) return;

    char* copy = strdup(s);
    char* tok = strtok(copy, ",");

    while (tok && *count < 16) {
        out[*count] = strtoll(tok, NULL, 10);
        (*count)++;
        tok = strtok(NULL, ",");
    }

    free(copy);
}

static int64_t egcd_(int64_t a, int64_t b, int64_t* x, int64_t* y) {
    if (a == 0) { *x = 0; *y = 1; return b; }
    int64_t x1, y1;
    int64_t g = egcd_(b % a, a, &x1, &y1);
    *x = y1 - (b / a) * x1;
    *y = x1;
    return g;
}

static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t res = 0;
    a %= mod;
    while (b) {
        if (b & 1) res = (res + a) % mod;
        a = (a << 1) % mod;
        b >>= 1;
    }
    return res;
}

void crt_recover_bigint(BigInt* out, 
                        const uint64_t* shares, 
                        const uint64_t* moduli, 
                        size_t n, 
                        uint64_t p_mod_val)
{
    BigInt M, result;
    biOne(&M);
    biZero(&result);

    for (size_t i = 0; i < n; ++i) {
        BigInt m_i; char buf[32];
        snprintf(buf, sizeof(buf), "%llu", (unsigned long long)moduli[i]);
        biFromDec(&m_i, buf);
        BigInt tmp;
        biMul(&tmp, &M, &m_i);
        biCopy(&M, &tmp);
    }

    for (size_t i = 0; i < n; ++i) {
        BigInt Mi, Mi_mod, inv, share_big, tmp1, tmp2, tmp3;

        BigInt mod_i; char buf[32];
        snprintf(buf, sizeof(buf), "%llu", (unsigned long long)moduli[i]);
        biFromDec(&mod_i, buf);

        biDiv(&Mi, &M, &mod_i);
        biMod(&Mi_mod, &Mi, &mod_i);

        if (!biModInv(&inv, &Mi_mod, &mod_i)) {
            fprintf(stderr, "Error: modular inverse does not exist\n");
            biZero(out);
            return;
        }

        snprintf(buf, sizeof(buf), "%llu", (unsigned long long)shares[i]);
        biFromDec(&share_big, buf);

        biMul(&tmp1, &share_big, &inv);
        biMul(&tmp2, &tmp1, &Mi);
        biAdd(&tmp3, &result, &tmp2);
        biCopy(&result, &tmp3);
    }

    BigInt S, p_mod;
    biMod(&S, &result, &M);

    char buf[32];
    snprintf(buf, sizeof(buf), "%llu", (unsigned long long)p_mod_val);
    biFromDec(&p_mod, buf);

    biMod(out, &S, &p_mod);
}


static void decode_backtrack(
    const char* digits,
    size_t len,
    size_t pos,
    const char* alph[],
    size_t alph_len,
    char* out,
    size_t out_pos,
    FILE* fptr
) {
    if (pos == len) {
        out[out_pos] = '\0';
        fwrite(out, 1, out_pos, fptr);
        fwrite("\n", 1, 1, fptr);
        return;
    }

    if (pos + 1 < len) {
        int n2 = (digits[pos] - '0') * 10 + (digits[pos + 1] - '0');
        if (n2 >= 1 && n2 <= (int)alph_len) {
            const char* ch = alph[n2 - 1];
            size_t l = strlen(ch);
            memcpy(out + out_pos, ch, l);
            decode_backtrack(digits, len, pos + 2, alph, alph_len, out, out_pos + l, fptr);
        }
    }

    if (digits[pos] != '0') {
        int n1 = digits[pos] - '0';
        if (n1 >= 1 && n1 <= 9 && n1 <= (int)alph_len) {
            const char* ch = alph[n1 - 1];
            size_t l = strlen(ch);
            memcpy(out + out_pos, ch, l);
            decode_backtrack(digits, len, pos + 1, alph, alph_len, out, out_pos + l, fptr);
        }
    }
}

const char* asmuthEntry(const char* alph_str,
                        const char* encText,
                        const char* frag) {
    (void)encText;

    const char* alph[64];
    size_t alph_len = 0;
    char buf[8];
    const char* p = alph_str;
    while (*p) {
        int bytes = 1;
        unsigned char c = (unsigned char)p[0];
        if (c >= 0xC0) {
            if (c < 0xE0) bytes = 2;
            else if (c < 0xF0) bytes = 3;
            else bytes = 4;
        }
        memcpy(buf, p, bytes);
        buf[bytes] = '\0';
        alph[alph_len++] = strdup(buf);
        p += bytes;
    }

    uint64_t shares[3] = {0}, moduli[3] = {0}, p_mod = 0;
    if (!frag || !*frag) { fprintf(stderr,"Error: -frag required\n"); return "[frag error]"; }

    char* copy = strdup(frag);
    char* parts[3] = {0};
    int i = 0;
    char* tok = strtok(copy, "|");
    while (tok && i < 3) parts[i++] = tok, tok = strtok(NULL, "|");
    if (i != 3) { fprintf(stderr,"Error: incomplete frag\n"); free(copy); return "[frag error]"; }

    tok = strtok(parts[0], ","); i = 0;
    while (tok && i < 3) { shares[i++] = strtoull(tok,NULL,10); tok=strtok(NULL,","); }
    if (i != 3) { fprintf(stderr,"Error: expected 3 shares\n"); free(copy); return "[frag error]"; }

    tok = strtok(parts[1], ","); i = 0;
    while (tok && i < 3) { moduli[i++] = strtoull(tok,NULL,10); tok=strtok(NULL,","); }
    if (i != 3) { fprintf(stderr,"Error: expected 3 moduli\n"); free(copy); return "[frag error]"; }

    p_mod = strtoull(parts[2], NULL, 10);
    if (!p_mod) { fprintf(stderr,"Error: invalid p_mod\n"); free(copy); return "[frag error]"; }

    free(copy);

    BigInt secret_big;
    crt_recover_bigint(&secret_big, shares, moduli, 3, p_mod);

    char secret_dec[64];
    biToDecString(&secret_big, secret_dec, sizeof(secret_dec));
    printf("%s\n", secret_dec);

    char numbuf[64];
    snprintf(numbuf, sizeof(numbuf), "%s", secret_dec);

    static char fname[128];
    strcpy(fname,"asmuth-");
    if(!append_time_txt(fname,(int)sizeof(fname))) strcat(fname,"unknown.txt");

    FILE* fptr = fopen(fname,"wb");
    if(!fptr) return "[file error]";

    char outbuf[512];
    decode_backtrack(numbuf, strlen(numbuf), 0, alph, alph_len, outbuf, 0, fptr);

    fclose(fptr);
    for(size_t j=0;j<alph_len;++j) free((void*)alph[j]);

    return fname;
}
