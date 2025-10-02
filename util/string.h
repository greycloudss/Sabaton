#pragma once
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif
#define MAX_KEYS 5




//comma seperated values
static void parseCSV(const char* s, int* out, int* count) {
    int idx = 0;
    const char* p = s;

    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;

        if (*p == '\0') break;

        if (*p == '?') {                
            out[idx++] = -1;
            while (*p && *p != ',') p++;
            continue;
        }

        char* end;
        long v = strtol(p, &end, 10);  
        if (end == p) {                
            while (*p && *p != ',') p++;
        } else {
            out[idx++] = (int)v;
            p = end;
        }
    }
    *count = idx;
}
static void invertVector(const int* v, int* inv, int n) {
    for (int i = 0; i < n; i++) {
        inv[v[i]] = i;
    }
}

static int stoi(const char* string) {
	int sign = (*string=='-') ? -1 : 1;
	long n = 0;
	string += (*string == '+' || *string == '-') ? 1 : 0;
	while (*string >= '0' && *string <= '9') n = n * 10 + (*string++ - '0');
	return (int)(sign * n);
}

static int m_strlen(const char* str, int buffcap) {
	if (buffcap <= 0) return 0;
	int i = 0;
	while (i < buffcap && str[i]) ++i;
	return i;
}

static int slen(const char* s) {
    int i = 0;
    while (s[i]) ++i;
    return i;
}

static void i32_to_str(int v, char* out, int* len) {
    unsigned int u = (v < 0) ? (unsigned int)(-v) : (unsigned int)v;
    char rev[16];
    int n = 0;
    if (u == 0) { rev[n++] = '0'; } else { while (u) { rev[n++] = (char)('0' + (u % 10)); u /= 10; } }
    int k = 0;
    if (v < 0) out[k++] = '-';
    for (int i = 0; i < n; ++i) out[k++] = rev[n - 1 - i];
    out[k] = '\0';
    *len = k;
}

static char* append_time_txt(char* s, int cap) {
    int n = slen(s);
    if (n >= cap) return NULL;
    time_t t = time(NULL);
    if (t != (time_t)-1) {
        int r = snprintf(s + n, (size_t)(cap - n), "%lld.txt", (long long)t);
        if (r > 0 && r < cap - n) return s;
    }
    const char* fb = "unknown.txt";
    int k = 0;
    while (fb[k]) ++k;
    if (n + k >= cap) return NULL;
    for (int i = 0; i <= k; ++i) s[n + i] = fb[i];
    return s;
}

static void largeWrite(const char* fname) {
    printf("[INFO] too many hits to output to stdout. Output - %s", fname);
}

/* UTF-8 helpers */

static int utf8_decode_one(const char* s, uint32_t* cp) {
    unsigned char c0 = (unsigned char)s[0];
    if (c0 < 0x80) { *cp = c0; return c0 ? 1 : 0; }
    if ((c0 >> 5) == 0x6) {
        unsigned char c1 = (unsigned char)s[1];
        if ((c1 & 0xC0) != 0x80) return 0;
        *cp = ((uint32_t)(c0 & 0x1F) << 6) | (uint32_t)(c1 & 0x3F);
        return 2;
    }
    if ((c0 >> 4) == 0xE) {
        unsigned char c1 = (unsigned char)s[1], c2 = (unsigned char)s[2];
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80)) return 0;
        *cp = ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)(c1 & 0x3F) << 6) | (uint32_t)(c2 & 0x3F);
        return 3;
    }
    if ((c0 >> 3) == 0x1E) {
        unsigned char c1 = (unsigned char)s[1], c2 = (unsigned char)s[2], c3 = (unsigned char)s[3];
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80) || ((c3 & 0xC0) != 0x80)) return 0;
        *cp = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(c1 & 0x3F) << 12) | ((uint32_t)(c2 & 0x3F) << 6) | (uint32_t)(c3 & 0x3F);
        return 4;
    }
    return 0;
}

static int utf8_encode_one(uint32_t cp, char out[4]) {
    if (cp <= 0x7F) { out[0] = (char)cp; return 1; }
    if (cp <= 0x7FF) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp <= 0xFFFF) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F)); out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static int utf8_to_u32(const char* s, uint32_t* out, int cap) {
    int n = 0;
    const char* p = s;
    while (*p && n < cap) {
        uint32_t cp; int adv = utf8_decode_one(p, &cp);
        if (adv <= 0) break;
        out[n++] = cp;
        p += adv;
    }
    return n;
}

static int u32_index_of(const uint32_t* alph, int m, uint32_t cp) {
    for (int i = 0; i < m; ++i) if (alph[i] == cp) return i;
    return -1;
}

static int u32_to_utf8(const uint32_t* cps, int n, char* out, int cap) {
    int k = 0;
    for (int i = 0; i < n; ++i) {
        char buf[4]; int len = utf8_encode_one(cps[i], buf);
        if (k + len >= cap) break;
        for (int j = 0; j < len; ++j) out[k++] = buf[j];
    }
    if (k < cap) out[k] = '\0';
    return k;
}

static void print(const char* s) {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD m;
    if (h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &m)) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
        if (wlen > 0) {
            WCHAR* w = (WCHAR*)malloc((size_t)wlen * sizeof(WCHAR));
            if (!w) return;
            MultiByteToWideChar(CP_UTF8, 0, s, -1, w, wlen);
            DWORD wr; WriteConsoleW(h, w, (DWORD)(wlen - 1), &wr, NULL);
            WriteConsoleW(h, L"\r\n", 2, &wr, NULL);
            free(w);
            return;
        }
    }
#endif
    fwrite(s, 1, slen(s), stdout);
    fwrite("\n", 1, 1, stdout);
}

static int parse_frag(const char* s, int* out, int cap){
    int n = 0, i = 0, sign = 1, acc = 0, innum = 0;
    while (s[i]){
        char c = s[i++];
        if (c == '-' && !innum){ sign = -1; innum = 1; acc = 0; }
        else if (c >= '0' && c <= '9'){ acc = acc*10 + (c - '0'); innum = 1; }
        else if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == '\r'){
            if (innum){
                if (n < cap) out[n++] = sign*acc;
                sign = 1; acc = 0; innum = 0;
            }
        } else {
            if (innum){
                if (n < cap) out[n++] = sign*acc;
                sign = 1; acc = 0; innum = 0;
            }
        }
    }
    if (innum && n < cap) out[n++] = sign*acc;
    return n;
}

static int* parse_frag_array(const char* s, int* out_n){
    int n = 0, i = 0, sign = 1, acc = 0, innum = 0;
    while (s[i]){
        char c = s[i++];
        if (c == '?'){
            if (innum){
                n++;
                innum = 0;
                sign = 1;
                acc = 0;
            }
            n++;
        } else if (c == '-' && !innum){
            sign = -1;
            innum = 1;
            acc = 0;
        } else if (c >= '0' && c <= '9'){
            acc = acc*10 + (c - '0');
            innum = 1;
        } else {
            if (innum){
                n++;
                innum = 0;
                sign = 1;
                acc = 0;
            }
        }
    }
    if (innum){
        n++;
    }

    int* out = (int*)malloc((size_t)n * sizeof(int));
    if (!out){
        if (out_n) *out_n = 0;
        return NULL;
    }
    if (out_n){
        *out_n = n;
    }

    i = 0;
    int k = 0;
    sign = 1;
    acc = 0;
    innum = 0;
    while (s[i] && k < n){
        char c = s[i++];
        if (c == '?'){
            if (innum){
                out[k++] = sign*acc;
                innum = 0;
                sign = 1;
                acc = 0;
            }
            out[k++] = -1;
        } else if (c == '-' && !innum){
            sign = -1;
            innum = 1;
            acc = 0;
        } else if (c >= '0' && c <= '9'){
            acc = acc*10 + (c - '0');
            innum = 1;
        } else {
            if (innum){
                out[k++] = sign*acc;
                innum = 0;
                sign = 1;
                acc = 0;
            }
        }
    }
    if (innum && k < n){
        out[k++] = sign*acc;
    }
    return out;
}

static void feistel_extract(const char* frag, char* out_flag, char** out_keys){
    *out_flag = 0;
    *out_keys = NULL;

    if (!frag || !*frag) return;

    const char* fpos = strstr(frag, "f=");

    if (fpos) *out_flag = (char)stoi(fpos + 2);

    const char* lb = strchr(frag, '[');
    const char* rb = NULL;

    if (lb) {
        const char* p = lb;
        while (*p && *p != ']') ++p;
        if (*p == ']') rb = p;
    }

    if (lb && rb && rb > lb) {
        int len = (int)(rb - lb + 1);
        char* buf = (char*)malloc((size_t)len + 1);
        for (int i = 0; i < len; ++i) buf[i] = lb[i];
        buf[len] = '\0';
        *out_keys = buf;
    } else {
        int len = slen(frag);
        char* buf = (char*)malloc((size_t)len + 1);
        for (int i = 0; i < len; ++i) buf[i] = frag[i];
        buf[len] = '\0';
        *out_keys = buf;
    }
}