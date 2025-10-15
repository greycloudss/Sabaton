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


const char* aesEntry(const char* alph, const char* encText, const char* frag);


#endif