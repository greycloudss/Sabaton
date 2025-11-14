#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "lithuanian.h"

#define BUF_INIT 4096

static char* read_line(FILE* f, char** buf, size_t* cap, size_t* out_len) {
    size_t used = 0;
    for (;;) {
        if (used + 2 >= *cap) {
            size_t ncap = (*cap) * 2;
            char* tmp = realloc(*buf, ncap);
            if (!tmp) return NULL;
            *buf = tmp;
            *cap = ncap;
        }
        if (!fgets(*buf + used, (int)(*cap - used), f)) {
            if (used == 0) return NULL;
            (*buf)[used] = '\0';
            break;
        }
        used += strlen(*buf + used);
        if (used && (*buf)[used - 1] == '\n') {
            (*buf)[used - 1] = '\0';
            used -= 1;
            break;
        }
    }
    char* out = malloc(used + 1);
    if (!out) return NULL;
    memcpy(out, *buf, used + 1);
    if (out_len) *out_len = used;
    return out;
}

const char* getExtension(const char* filename) {
    const char* dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}

float d_voice_measure(const char* line, size_t len) {
    static const char* digs[] = {
        "al","am","an","ar","el","em","en","er",
        "il","im","in","ir","ul","um","un","ur",
        "ai","au","ei","ui","ie","uo"
    };
    if (len < 2) return 0.0f;
    int count = 0;
    for (size_t i = 0; i + 1 < len; ++i) {
        char a = (char)tolower((unsigned char)line[i]);
        char b = (char)tolower((unsigned char)line[i+1]);
        for (size_t k = 0; k < 22; ++k) {
            if (a == digs[k][0] && b == digs[k][1]) {
                count++;
                break;
            }
        }
    }
    return len ? (float)count / (float)len : 0.0f;
}

typedef struct { int idx; float score; } Row;

static int cmp_desc(const void* a, const void* b) {
    const Row* ra = (const Row*)a;
    const Row* rb = (const Row*)b;
    if (ra->score < rb->score) return 1;
    if (ra->score > rb->score) return -1;
    return ra->idx - rb->idx;
}

static void process_and_write(FILE* out, FILE* in) {
    size_t bcap = BUF_INIT;
    char* buf = malloc(bcap);
    if (!buf) return;

    size_t lcap = 64, rcap = 64;
    char** lines = malloc(lcap * sizeof(char*));
    Row* rows = malloc(rcap * sizeof(Row));
    if (!lines || !rows) {
        free(buf);
        free(lines);
        free(rows);
        return;
    }

    int n = 0;
    for (;;) {
        size_t llen = 0;
        char* line = read_line(in, &buf, &bcap, &llen);
        if (!line) break;
        if ((size_t)n == lcap) {
            size_t nl = lcap * 2;
            char** t1 = realloc(lines, nl * sizeof(char*));
            if (!t1) {
                free(line);
                break;
            }
            lines = t1; lcap = nl;
        }
        if ((size_t)n == rcap) {
            size_t nr = rcap * 2;
            Row* t2 = realloc(rows, nr * sizeof(Row));
            if (!t2) {
                free(line);
                break;
            }
            rows = t2; rcap = nr;
        }
        lines[n] = line;
        rows[n].idx = n;
        rows[n].score = d_voice_measure(line, llen);
        n++;
    }

    if (n > 0) {
        qsort(rows, (size_t)n, sizeof(Row), cmp_desc);
        for (int i = 0; i < n; ++i) {
            fputs(lines[rows[i].idx], out);
            fputc('\n', out);
        }
    }

    for (int i = 0; i < n; ++i) free(lines[i]);
    free(rows);
    free(lines);
    free(buf);
}

const char* recognEntry(const char* bruteFile) {
    if (!bruteFile) return NULL;

    FILE* in = fopen(bruteFile, "rb");
    if (!in) return NULL;

    size_t bflen = strlen(bruteFile);
    size_t alloc = bflen + 5;
    char* enhFname = malloc(alloc);
    if (!enhFname) {
        fclose(in);
        return NULL;
    }
    snprintf(enhFname, alloc, "enh-%s", bruteFile);

    FILE* out = fopen(enhFname, "wb");
    if (!out) {
        free(enhFname);
        fclose(in);
        return NULL;
    }

    process_and_write(out, in);

    fclose(in);
    fclose(out);
    return enhFname;
}
