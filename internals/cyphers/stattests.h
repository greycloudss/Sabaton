#define STATTESTS_H

#include <stddef.h>

#ifdef __cplusplus
    extern "C" {
#endif

const char* statEntry(const char* alph, const char* encText);

void bitTest(const unsigned char* bits, int n, double* T, double* p);
void pairTest(const unsigned char* bits, int n, double* T, double* p);
void pokerTest(const unsigned char* bits, int n, int m, double* T, double* p);
void autoCorrelationTest(const unsigned char* bits, int n, int d, double* T, double* p);

#ifdef __cplusplus
}
#endif