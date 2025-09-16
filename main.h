#pragma once
#include <string.h>

//cyphers
#include "internals/cyphers/affineCaesar.h"

#define FLAG_COUNT 4

extern volatile char killswitch;

typedef struct {
    char flags[FLAG_COUNT];
    const char* wordlist;
    unsigned char minLength;
    unsigned char maxLength;
    const char* specialRegex;

    char decrypt;
    char decypher;
    char affineCaesar;
    char brute;
    const char* frag;
    const char* alph;

    const char* encText;
    const char* out;
} Arguments;