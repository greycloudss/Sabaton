#pragma once

#include "string.h"  
#include "number.h" 
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    char *key;
    char *value;
} FragPair;

typedef struct {
    FragPair *items;
    size_t count;
} FragMap;


static inline char *fragStrdupRange(const char *start, const char *end) {
    if (!start || !end || end < start) return NULL;
    size_t len = (size_t)(end - start);
    char *s = (char *)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, start, len);
    s[len] = '\0';
    return s;
}

static inline void fragTrimInplace(char *s) {
    if (!s) return;
    char *p = s;
    while (*p && isspace((unsigned char)*p)) p++;
    if (p != s) memmove(s, p, strlen(p) + 1);

    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[--len] = '\0';
    }
}


static inline char **fragTokensSplit(const char *s, char delim, size_t *out_count) {
    if (out_count) *out_count = 0;
    if (!s) return NULL;

    size_t count = 1;
    for (const char *p = s; *p; ++p) {
        if (*p == delim) count++;
    }

    char **arr = (char **)malloc(count * sizeof(char *));
    if (!arr) return NULL;

    size_t idx = 0;
    const char *start = s;
    const char *p = s;
    for (;; ++p) {
        if (*p == delim || *p == '\0') {
            arr[idx++] = fragStrdupRange(start, p);
            if (*p == '\0') break;
            start = p + 1;
        }
    }

    if (out_count) *out_count = idx;
    return arr;
}

static inline void fragTokensFree(char **tokens, size_t count) {
    if (!tokens) return;
    for (size_t i = 0; i < count; ++i) {
        free(tokens[i]);
    }
    free(tokens);
}


static inline FragMap fragmapParse(const char *frag,
                                   char pair_delim,
                                   char kv_delim1,
                                   char kv_delim2)
{
    FragMap map = { NULL, 0 };
    if (!frag || !*frag) return map;

    size_t tok_count = 0;
    char **tokens = fragTokensSplit(frag, pair_delim, &tok_count);
    if (!tokens || tok_count == 0) {
        fragTokensFree(tokens, tok_count);
        return map;
    }

    map.items = (FragPair *)calloc(tok_count, sizeof(FragPair));
    if (!map.items) {
        fragTokensFree(tokens, tok_count);
        return map;
    }

    size_t used = 0;

    for (size_t i = 0; i < tok_count; ++i) {
        char *t = tokens[i];
        if (!t) continue;
        fragTrimInplace(t);
        if (!*t) {
            free(t);
            continue;
        }

        char *sep = strchr(t, kv_delim1);
        if (!sep && kv_delim2) sep = strchr(t, kv_delim2);
        if (!sep) {
            map.items[used].key = fragStrdupRange(t, t + strlen(t));
            map.items[used].value = NULL;
            used++;
            free(t);
            continue;
        }

        char *key = fragStrdupRange(t, sep);
        char *val = fragStrdupRange(sep + 1, t + strlen(t));
        free(t);

        if (!key || !val) {
            free(key);
            free(val);
            continue;
        }

        fragTrimInplace(key);
        fragTrimInplace(val);

        map.items[used].key = key;
        map.items[used].value = val;
        used++;
    }

    fragTokensFree(tokens, tok_count);
    map.count = used;
    return map;
}

static inline const char *fragmapGet(const FragMap *map, const char *key) {
    if (!map || !key) return NULL;
    for (size_t i = 0; i < map->count; ++i) {
        if (map->items[i].key && strcmp(map->items[i].key, key) == 0) {
            return map->items[i].value;
        }
    }
    return NULL;
}

static inline void fragmapFree(FragMap *map) {
    if (!map || !map->items) return;
    for (size_t i = 0; i < map->count; ++i) {
        free(map->items[i].key);
        free(map->items[i].value);
    }
    free(map->items);
    map->items = NULL;
    map->count = 0;
}


static inline size_t fragParseIntList(const char *str,
                                      long long *out,
                                      size_t max_out)
{
    if (!str || !out || max_out == 0) return 0;
    int count = 0;
    unsigned long long *tmp = parse_ull_array(str, &count);
    if (!tmp) return 0;

    size_t n = (size_t)((count < (int)max_out) ? count : (int)max_out);
    for (size_t i = 0; i < n; ++i) {
        out[i] = (long long)tmp[i];
    }
    free(tmp);
    return n;
}

static inline int fragSplitPair(const char *frag,
                                char sep,
                                char **left,
                                char **right)
{
    if (!frag || !left || !right) return 0;
    const char *p = strchr(frag, sep);
    if (!p) return 0;

    *left  = fragStrdupRange(frag, p);
    *right = fragStrdupRange(p + 1, frag + strlen(frag));
    if (!*left || !*right) {
        free(*left);
        free(*right);
        *left = *right = NULL;
        return 0;
    }
    fragTrimInplace(*left);
    fragTrimInplace(*right);
    return 1;
}

static inline const char *fragGetScalar(const FragMap *map, const char *key) {
    return fragmapGet(map, key);   
}


static inline long long fragGetLongLong(const FragMap *map,
                                        const char *key,
                                        long long def,
                                        int *ok_out)
{
    if (ok_out) *ok_out = 0;
    const char *v = fragmapGet(map, key);
    if (!v) return def;
    char *end = NULL;
    long long x = strtoll(v, &end, 10);
    if (end == v) return def;
    if (ok_out) *ok_out = 1;
    return x;
}


static inline size_t fragGetNumArray(const FragMap *map,
                                     const char *key,
                                     long long *out,
                                     size_t max_out)
{
    const char *v = fragmapGet(map, key);
    if (!v) return 0;
    return fragParseIntList(v, out, max_out);
}


static inline char **fragGetStringArray(const FragMap *map,
                                        const char *key,
                                        size_t *out_count)
{
    if (out_count) *out_count = 0;
    const char *v = fragmapGet(map, key);
    if (!v || !*v) return NULL;

    const char *start = v;
    const char *end   = v + strlen(v);
    while (start < end && isspace((unsigned char)*start)) start++;
    while (end > start && isspace((unsigned char)end[-1])) end--;

    if (end > start && *start == '[' && end[-1] == ']') {
        start++; end--;
        while (start < end && isspace((unsigned char)*start)) start++;
        while (end > start && isspace((unsigned char)end[-1])) end--;
    }

    char *inner = fragStrdupRange(start, end);
    if (!inner) return NULL;

    size_t cnt = 0;
    char **parts = fragTokensSplit(inner, ',', &cnt);
    free(inner);
    if (!parts || cnt == 0) {
        fragTokensFree(parts, cnt);
        return NULL;
    }

    for (size_t i = 0; i < cnt; ++i) {
        fragTrimInplace(parts[i]);
        char *s = parts[i];
        size_t len = strlen(s);
        if (len >= 2 && s[0] == '"' && s[len - 1] == '"') {
            s[len - 1] = '\0';
            memmove(s, s + 1, len - 1);
        }
    }

    if (out_count) *out_count = cnt;
    return parts;  
}


// bet kokiam seperator :, =, | ir t.t
static inline FragMap fragmapParseTupleSep(const char *expr, char sep)
{
    FragMap map = (FragMap){ NULL, 0 };
    if (!expr || !*expr) return map;

    char *left = NULL;
    char *right = NULL;
    if (!fragSplitPair(expr, sep, &left, &right)) {
        return map;
    }

    const char *l_start = left;
    const char *l_end   = left + strlen(left);
    while (l_start < l_end && isspace((unsigned char)*l_start)) l_start++;
    while (l_end > l_start && isspace((unsigned char)l_end[-1])) l_end--;

    if (l_end > l_start && *l_start == '[' && l_end[-1] == ']') {
        l_start++; l_end--;
        while (l_start < l_end && isspace((unsigned char)*l_start)) l_start++;
        while (l_end > l_start && isspace((unsigned char)l_end[-1])) l_end--;
    }

    const char *r_start = right;
    const char *r_end   = right + strlen(right);
    while (r_start < r_end && isspace((unsigned char)*r_start)) r_start++;
    while (r_end > r_start && isspace((unsigned char)r_end[-1])) r_end--;

    if (r_end > r_start && *r_start == '[' && r_end[-1] == ']') {
        r_start++; r_end--;
        while (r_start < r_end && isspace((unsigned char)*r_start)) r_start++;
        while (r_end > r_start && isspace((unsigned char)r_end[-1])) r_end--;
    }

    char *left_inner  = fragStrdupRange(l_start,  l_end);
    char *right_inner = fragStrdupRange(r_start, r_end);
    free(left);
    free(right);
    if (!left_inner || !right_inner) {
        free(left_inner);
        free(right_inner);
        return map;
    }

    size_t key_count = 0;
    char **keys = fragTokensSplit(left_inner, ',', &key_count);
    size_t val_count = 0;
    char **vals = fragTokensSplit(right_inner, ',', &val_count);
    free(left_inner);
    free(right_inner);

    if (!keys || !vals || key_count == 0 || val_count == 0) {
        fragTokensFree(keys, key_count);
        fragTokensFree(vals, val_count);
        return map;
    }

    size_t count = (key_count < val_count) ? key_count : val_count;
    map.items = (FragPair *)calloc(count, sizeof(FragPair));
    if (!map.items) {
        fragTokensFree(keys, key_count);
        fragTokensFree(vals, val_count);
        return map;
    }

    size_t used = 0;
    for (size_t i = 0; i < count; ++i) {
        char *k = keys[i];
        char *v = vals[i];
        if (!k || !v) continue;

        fragTrimInplace(k);
        fragTrimInplace(v);

        size_t klen = strlen(k);
        if (klen >= 2 && k[0] == '"' && k[klen - 1] == '"') {
            k[klen - 1] = '\0';
            memmove(k, k + 1, klen - 1);
        }
        size_t vlen = strlen(v);
        if (vlen >= 2 && v[0] == '"' && v[vlen - 1] == '"') {
            v[vlen - 1] = '\0';
            memmove(v, v + 1, vlen - 1);
        }

        map.items[used].key   = k;   
        map.items[used].value = v;
        used++;
    }

    free(keys);
    free(vals);

    map.count = used;
    return map;
}
