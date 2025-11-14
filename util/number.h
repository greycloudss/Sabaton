#pragma once
#define __INT_MIN (-2147483647 - 1)
#define __INT_MAX 2147483647

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

static inline unsigned long uabs_l(long x){
    unsigned long ux = (unsigned long)x;
    unsigned long m  = (unsigned long)-(long)(ux >> (sizeof(long)*8 - 1));
    return (ux ^ m) - m;
}

inline static unsigned long gcd(unsigned long x, unsigned  long y) {
    #ifdef __x86_64__
        // IF SOMETHING BREAKS UNCOMMENT THIS BLOCK AND COMMENT THE ONE WITH ASM
        /* 
            if (!x) return y;
            if (!y) return x;
            unsigned k = __builtin_ctzl(x | y);
            x >>= __builtin_ctzl(x);
            do{
                y >>= __builtin_ctzl(y);
                if (x > y){
                    unsigned long t = x;
                    x = y;
                    y = x;
                }
                y -= x;
            } while (y);
            return x << k;
        */
        //unsigned long x = uabs_l(xa), y = uabs_l(ya);
        if (!x) return y;
        if (!y) return x;

        unsigned k = __builtin_ctzl(x | y);
        if ((x & 1ul) == 0) x >>= __builtin_ctzl(x);
        
        do {
           if ((y & 1ul) == 0) y >>= __builtin_ctzl(y);
            __asm__ volatile(
                ".intel_syntax noprefix\n\t"
                "cmp %0, %1\n\t"
                "jle 1f\n\t"
                "xchg %0, %1\n\t"
                "1:\n\t"
                "sub %1, %0\n\t"
                ".att_syntax prefix"
                :"+r"(x), "+r"(y)
                :
                : "cc"
            );
        } while (y);
        return x << k;
    #else
        int result = ((xa < ya) ? xa : ya);
        while (result > 0) {
            if (a % result == 0 && b % result == 0) {
                break;
            }
            result--;
        }
        return result;
    #endif
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

static void swap_int(int* a, int* b) { // f u my xor swap is better cuz no random var
    if (a == b) return;
    *a ^= *b;
    *b ^= *a;
    *a ^= *b;
}

static void mat2_mul_vec_mod(const int k[4], const int v[2], int m, int out[2]){
    out[0] = mod(k[0]*v[0] + k[1]*v[1], m);
    out[1] = mod(k[2]*v[0] + k[3]*v[1], m);
}
