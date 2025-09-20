#define INT_MIN (-2147483647 - 1)
#define INT_MAX 2147483647

static int iPow(int x, int power) {
    if (power < 0) return 0;
    long long r = 1, b = x;
    int e = power;
    while (e) {
        if (e & 1) {
            r *= b;
            if (r > INT_MAX) return INT_MAX;
            if (r < INT_MIN) return INT_MIN;
        }
        e >>= 1;
        if (e) {
            b *= b;
            if (b > INT_MAX) b = (long long)INT_MAX + 1;
            if (b < INT_MIN) b = (long long)INT_MIN - 1;
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