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
