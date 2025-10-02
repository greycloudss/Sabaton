#include "feistel.h"
#include "../lithuanian.h"

#define MAX_FUNCS 5

static char* g_first_line = NULL;
static size_t g_first_len = 0;
static int g_stop = 0;

static int is_valid_plaintext(const char* s){
    if (!s) return 0;
    for (const unsigned char* p = (const unsigned char*)s; *p; ++p){
        if (!(*p == ' ' || (*p >= 'A' && *p <= 'Z'))) return 0;
    }
    return 1;
}

int selFunc(char flag, uint8_t m, uint8_t k){
    switch (flag){
        case 0: return (uint8_t)((m | k) ^ ((k / 16) & m));
        case 1: return (uint8_t)((m & k) | ((k % 16) ^ m));
        case 2: return (uint8_t)((m | k) ^ ((k / 16) & m));
        case 3: return (uint8_t)((m ^ k) & ((k / 16) | m));
        default: return (uint8_t)((m & k) ^ ((k % 16) | m));
    }
}

char* pieceFesitel(const char* encText, int* keys, size_t n, char funcFlag){
    int bigN = 0;
    int* encInt = parse_frag_array(encText, &bigN);
    if (!encInt || bigN <= 0 || (bigN & 1)){
        if (encInt) free(encInt);
        return NULL;
    }

    char* ans = (char*)malloc((size_t)bigN + 1);
    if (!ans){
        free(encInt);
        return NULL;
    }

    int pos = 0;
    for (int i = 0; i < bigN; i += 2){
        uint8_t R = (uint8_t)encInt[i];
        uint8_t L = (uint8_t)encInt[i + 1];
        for (int j = (int)n - 1; j >= 0; --j){
            uint8_t t = (uint8_t)(R ^ selFunc(funcFlag, L, (uint8_t)keys[j]));
            R = L;
            L = t;
        }
        ans[pos++] = (char)L;
        ans[pos++] = (char)R;
    }
    for (int i = 0; i < bigN; ++i){
        if (ans[i] == '\0') ans[i] = ' ';
    }
    ans[pos] = '\0';

    free(encInt);
    return ans;
}

static void write_candidate_if_valid_and_check_stop(FILE* fptr, char* a){
    if (!a) return;
    if (!is_valid_plaintext(a)){
        free(a);
        return;
    }
    size_t len = strlen(a);
    if (!g_first_line){
        g_first_line = (char*)malloc(len + 1);
        if (g_first_line){
            memcpy(g_first_line, a, len + 1);
            g_first_len = len;
        }
        fwrite(a, 1, len, fptr);
        fwrite("\n", 1, 1, fptr);
    } else {
        if (len == g_first_len && memcmp(a, g_first_line, len) == 0){
            free(a);
            g_stop = 1;
            return;
        }
        fwrite(a, 1, len, fptr);
        fwrite("\n", 1, 1, fptr);
    }
    free(a);
}

static void recursiveGenerator(int depth, size_t n, const char* encText, int* frag, FILE* fptr, char flag){
    if (g_stop) return;
    int place = -1;
    for (size_t i = 0; i < n; ++i){
        if (frag[i] == -1){
            place = (int)i;
            break;
        }
    }
    if (place == -1 || depth == 0){
        if ((unsigned char)flag >= MAX_FUNCS){
            for (int j = 0; j < MAX_FUNCS && !g_stop; ++j){
                char* a = pieceFesitel(encText, frag, n, (char)j);
                write_candidate_if_valid_and_check_stop(fptr, a);
            }
        } else {
            if (!g_stop){
                char* a = pieceFesitel(encText, frag, n, flag);
                write_candidate_if_valid_and_check_stop(fptr, a);
            }
        }
        return;
    }
    for (int v = 0; v < 256; ++v){
        if (g_stop) break;
        frag[place] = v;
        recursiveGenerator(depth - 1, n, encText, frag, fptr, flag);
    }
    frag[place] = -1;
}

const char* partialFeistel(const char* encText, int* frag, size_t n, char flag){
    int missing = 0;
    for (size_t i = 0; i < n; ++i){
        if (frag[i] == -1) missing++;
    }

    static char fname[128];
    const char* base = "feistel-";
    int p = 0;
    while (base[p] && p < (int)sizeof(fname) - 1){
        fname[p] = base[p];
        ++p;
    }
    fname[p] = '\0';
    if (!append_time_txt(fname, (int)sizeof fname)){
        const char* fb = "unknown.txt";
        int i = 0;
        while (fb[i] && p + i < (int)sizeof(fname) - 1){
            fname[p + i] = fb[i];
            ++i;
        }
        fname[p + i] = '\0';
    }

    FILE* fptr = fopen(fname, "wb");
    if (!fptr) return "";

    g_stop = 0;
    if (g_first_line){
        free(g_first_line);
        g_first_line = NULL;
        g_first_len = 0;
    }

    if (missing == 0){
        recursiveGenerator(0, n, encText, frag, fptr, flag);
    } else {
        recursiveGenerator(missing, n, encText, frag, fptr, flag);
    }

    if (g_first_line){
        free(g_first_line);
        g_first_line = NULL;
        g_first_len = 0;
    }

    fclose(fptr);
    return recognEntry(fname);
}

const char* bruteFeistel(const char* encText, int* frag, char flag){
    int keys[3] = { -1, -1, -1 };
    if (frag){
        for (int i = 0; i < 3; ++i){
            keys[i] = frag[i];
        }
    }
    return recognEntry(partialFeistel(encText, keys, 3, flag));
}

const char* feistelEntry(const char* encText, const char* frag, char flag){
    if (!frag || !*frag) return bruteFeistel(encText, NULL, flag);

    int n = 0;
    int* keys = parse_frag_array(frag, &n);
    if (!keys) return bruteFeistel(encText, NULL, flag);

    int has_missing = 0;
    for (int i = 0; i < n; ++i){
        if (keys[i] == -1){
            has_missing = 1;
            break;
        }
    }

    if (has_missing){
        const char* res = partialFeistel(encText, keys, (size_t)n, flag);
        free(keys);
        return res;
    } else {
        char* out = pieceFesitel(encText, keys, (size_t)n, flag);
        free(keys);
        return out ? out : "";
    }
}
