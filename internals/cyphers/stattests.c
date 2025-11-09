#include "stattests.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char* textToBits(const char* text, size_t* outLen) {
    if (!text) return NULL;
    size_t len = strlen(text);
    unsigned char* bits = malloc(len * 8);
    if (!bits) return NULL;
    size_t k = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        for (int b = 7; b >= 0; b--)
            bits[k++] = (c >> b) & 1;
    }
    if (outLen) *outLen = k;
    return bits;
}

void bitTest(const unsigned char* bits, int n, double* T, double* p) {
    if (!bits || n <= 0) return;
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += bits[i] ? 1 : -1;

    *T = fabs(sum) / sqrt(n);
    *p = erfc(*T / sqrt(2.0));
}

void pairTest(const unsigned char* bits, int n, double* T, double* p) {
    if (!bits || n < 2) return;
    int counts[4] = {0};
    int pairs = n / 2;
    for (int i = 0; i < pairs; i++) {
        int b0 = bits[2*i];
        int b1 = bits[2*i + 1];
        int idx = (b0 << 1) | b1;
        counts[idx]++;
    }

    double chi2 = 0.0;
    double expected = pairs / 4.0;
    for (int i = 0; i < 4; i++) {
        double diff = counts[i] - expected;
        chi2 += diff * diff / expected;
    }

    *T = chi2;
    *p = exp(-0.5 * chi2);
}

void pokerTest(const unsigned char* bits, int n, int m, double* T, double* p) {
    if (!bits || m < 3 || m > 10) return;
    int groups = n / m;
    int maxVal = 1 << m;
    int* counts = calloc(maxVal, sizeof(int));
    if (!counts) return;

    for (int i = 0; i < groups; i++) {
        int val = 0;
        for (int j = 0; j < m; j++)
            val = (val << 1) | bits[i*m + j];
        counts[val]++;
    }

    double sum = 0;
    for (int i = 0; i < maxVal; i++)
        sum += counts[i] * counts[i];

    *T = ((double)maxVal / groups) * sum - groups;
    *p = exp(-0.5 * (*T));
    free(counts);
}

void autoCorrelationTest(const unsigned char* bits, int n, int d, double* T, double* p) {
    if (!bits || d < 1 || n <= d) return;
    int sum = 0;
    for (int i = 0; i < n - d; i++)
        sum += (bits[i] == bits[i + d]) ? 1 : 0;

    *T = 2.0 * (sum - (n - d) / 2.0) / sqrt(n - d);
    *p = erfc(fabs(*T) / sqrt(2.0));
}

const char* statEntry(const char* alph, const char* encText) {
    (void)alph;

    size_t n;
    unsigned char* bits = textToBits(encText, &n);
    if (!bits) return "Error: could not convert to bits.";

    static char buffer[1024];
    double T, p;

    char* out = buffer;
    out[0] = '\0';

    strcat(out, "Bit test:\n");
    bitTest(bits, n, &T, &p);
    sprintf(out + strlen(out), "  T = %.3f, p = %.3f, %s\n", T, p, (p >= 0.05 ? "H0 accepted" : "H0 rejected"));

    strcat(out, "Pair test:\n");
    pairTest(bits, n, &T, &p);
    sprintf(out + strlen(out), "  T = %.3f, p = %.3f, %s\n", T, p, (p >= 0.05 ? "H0 accepted" : "H0 rejected"));

    strcat(out, "Poker test (m=5):\n");
    pokerTest(bits, n, 5, &T, &p);
    sprintf(out + strlen(out), "  T = %.3f, p = %.3f, %s\n", T, p, (p >= 0.05 ? "H0 accepted" : "H0 rejected"));

    strcat(out, "Autocorrelation test (d=5):\n");
    autoCorrelationTest(bits, n, 5, &T, &p);
    sprintf(out + strlen(out), "  T = %.3f, p = %.3f, %s\n", T, p, (p >= 0.05 ? "H0 accepted" : "H0 rejected"));

    free(bits);
    return buffer;
}
