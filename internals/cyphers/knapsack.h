#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>



static unsigned long long mulmod_u64(unsigned long long a, unsigned long long b, unsigned long long m);

static long long egcd64(long long a, long long b, long long* x, long long* y);
static unsigned long long modinv_u64(unsigned long long a, unsigned long long m);

static unsigned long long* parse_ull_array(const char* s, int* outCount);

static int bits_to_byte_msb(const int* bits, int n);
void extractKnapsackValues(
        const char* frag,
        unsigned long long** keys_out, int* keyCount_out,
        unsigned long long* p_out,
        unsigned long long* w1_out);