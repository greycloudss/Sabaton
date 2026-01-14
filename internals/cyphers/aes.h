#ifndef AES_H
#define AES_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../util/string.h"
#include "../../util/number.h"


typedef struct {
    uint64_t key;
    int k1_val;
    int used;
} HashEntry;

typedef struct {
    HashEntry *table;
    size_t size;    
    size_t mask;    
} HashTable;

#ifdef __cplusplus
extern "C" {
#endif
char* decryptAESV(const int* cipher, int nBlocks, int p, int a, int b, const int T[4], const int K1[4], int rounds);
char* encryptAESV(const int* plain, int nBlocks, int p, int a, int b, const int T[4], const int K1[4], int rounds);
const char* aesEntry(const char* alph, const char* encText, const char* frag);
#ifdef __cplusplus
}
#endif

#endif
