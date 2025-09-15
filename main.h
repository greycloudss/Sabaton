#pragma once
#include <string.h>

//cyphers
#include "internals/hashes/affineCaesar.h"

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
} Arguments;

#include "afineCaesar.h"

void decypher(Arguments* args) {
    if (!args || !args->decypher) return;

    if (args->affineCaesar) {
        const char* frag = args->brute ? NULL : args->frag;
        const char* res = afineCaesarEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }
}


