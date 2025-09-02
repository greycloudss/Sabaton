#define INT_MIN (-2147483647 - 1)
#define INT_MAX 2147483647

int iPow(int x, int power) {
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
