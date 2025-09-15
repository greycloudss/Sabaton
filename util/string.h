#pragma once
#include <stdio.h>
#include <time.h>


static int stoi(const char* string) {
	int sign = (*string=='-') ? -1 : 1;
	long n = 0;
  
	string += (*string == '+' || *string == '-') ? 1 : 0;

	while (*string >= '0' && *string <= '9') n = n * 10 + (*string++ - '0');
  
	return (int)(sign * n);
}


static int m_strlen(const char* str, int buffcap) { //buff cap due to buffer overflow
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

    if (u == 0) {
        rev[n++] = '0';
    } else {
        while (u) {
            rev[n++] = (char)('0' + (u % 10));
            u /= 10;
        }
    }

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
