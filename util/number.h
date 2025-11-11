#pragma once

#include <stddef.h>
#include <stdlib.h>
#define __INT_MIN (-2147483647 - 1)
#define __INT_MAX 2147483647



static unsigned long long mulmod_u64(unsigned long long a, unsigned long long b, unsigned long long m){
#if defined(__SIZEOF_INT128__)
    __uint128_t x = (__uint128_t)(a % m) * (__uint128_t)(b % m);
    return (unsigned long long)(x % m);
#else
    unsigned long long res = 0ULL; a %= m; b %= m;
    while (b){
        if (b & 1ULL){ res = (res >= m - a) ? (res + a - m) : (res + a); }
        a = (a >= m - a) ? (a + a - m) : (a + a);
        b >>= 1ULL;
    }
    return res;
#endif
}

static long long egcd64(long long a, long long b, long long* x, long long* y){
    if (b == 0){ *x = 1; *y = 0; return a; }
    long long x1, y1; long long g = egcd64(b, a % b, &x1, &y1);
    *x = y1; *y = x1 - (a / b) * y1; return g;
}
static unsigned long long modinv_u64(unsigned long long a, unsigned long long m){
    long long x, y; long long g = egcd64((long long)(a % m), (long long)m, &x, &y);
    if (g != 1) return 0ULL;
    long long r = x % (long long)m; if (r < 0) r += (long long)m;
    return (unsigned long long)r;
}

static unsigned long long* parse_ull_array(const char* s, int* outCount) {
    if (outCount) *outCount = 0;
    if (!s) return NULL;

    int cap = 16;
    int count = 0;
    unsigned long long* arr = (unsigned long long*)malloc(sizeof(unsigned long long) * (size_t)cap);
    if (!arr) return NULL;

    unsigned long long acc = 0;
    int inNum = 0;

    for (const char* p = s; ; ++p) {
        int c = (unsigned char)*p;

        if (c >= '0' && c <= '9') {
            inNum = 1;
            acc = acc * 10ULL + (unsigned long long)(c - '0');
        } else {
            if (inNum) {
                if (count == cap) {
                    cap *= 2;
                    unsigned long long* tmp = (unsigned long long*)realloc(arr, sizeof(unsigned long long) * (size_t)cap);
                    if (!tmp) { free(arr); return NULL; }
                    arr = tmp;
                }
                arr[count++] = acc;
                acc = 0;
                inNum = 0;
            }
        }

        if (c == '\0') break; 
    }

    if (outCount) *outCount = count;
    return arr;
}

static int bits_to_byte_msb(const int* bits, int n){
    int v=0; for (int i=0;i<n;++i) v = (v<<1) | (bits[i] ? 1 : 0);
    return v & 0xFF;
}

static int iPow(int x, int power) {
    if (power < 0) return 0;
    long long r = 1, b = x;
    int e = power;
    while (e) {
        if (e & 1) {
            r *= b;
            if (r > __INT_MAX) return __INT_MAX;
            if (r < __INT_MIN) return __INT_MIN;
        }
        e >>= 1;
        if (e) {
            b *= b;
            if (b > __INT_MAX) b = (long long)__INT_MAX + 1;
            if (b < __INT_MIN) b = (long long)__INT_MIN - 1;
        }
    }
    return (int)r;
}


static char* numbersToBytes(const int* v, size_t n) {
    char* s = malloc(n + 1);
    if (!s) return NULL;
    for (size_t i = 0; i < n; ++i) {
        int x = v[i];
        if (x < 0 || x > 255) s[i] = '?';
        else s[i] = (char)(unsigned char)x;
    }
    s[n] = '\0';
    return s;
}


static int isPrime(int n) {
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    for (int i = 3; i * i <= n; i += 2)
        if (n % i == 0) return 0;
    return 1;
}

static int modmul(long a, long b, int p) {
    long r = (a % p) * (b % p);
    return (int)(r % p);
}


static int mod(int a, int m) {
    int r = a % m;
    return r < 0 ? r + m : r;
}

//straight up yoinked from geeksforgeeks
static int gcd(int a, int b) {
    int result = ((a < b) ? a : b);
    while (result > 0) {
        if (a % result == 0 && b % result == 0) {
            break;
        }
        result--;
    }
    return result;
}


/*
this is the extended Euclidean algorithm which basically finds a gcd between a and b but also gets two integers
x and y such that ax + by = gcd(a, b).
*/

static int egcd(int a, int b, int* x, int* y) {
    if (b == 0) { *x = 1; *y = 0; return a; }
    int x1, y1;
    int g = egcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

static int modinv(int a, int m, int* inv) {
    int x, y;
    int g = egcd(mod(a, m), m, &x, &y);
    if (g != 1) return 0;
    *inv = mod(x, m);
    return 1;
}

static int det2_mod(const int k[4], int m){
    int d = k[0]*k[3] - k[1]*k[2];
    return mod(d, m);
}

static int inv2x2mod(const int K[4], int m, int invK[4]) {
    long a = K[0], b = K[1], c = K[2], d = K[3];
    long det = (a*d - b*c);
    int detm = mod((int)det, m);

    int detInv;
    if (!modinv(detm, m, &detInv)) return 0;

    long A =  d, B = -b, C = -c, D =  a;
    invK[0] = mod((int)(detInv * A), m);
    invK[1] = mod((int)(detInv * B), m);
    invK[2] = mod((int)(detInv * C), m);
    invK[3] = mod((int)(detInv * D), m);
    return 1;
}


static void mat2_mul_vec_mod(const int k[4], const int v[2], int m, int out[2]){
    out[0] = mod(k[0]*v[0] + k[1]*v[1], m);
    out[1] = mod(k[2]*v[0] + k[3]*v[1], m);
}
