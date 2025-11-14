#pragma once
#include <string.h>

// cyphers
#include "internals/cyphers/affineCaesar.h"
#include "internals/cyphers/hill.h"
#include "internals/cyphers/vigenere.h"
#include "internals/cyphers/feistel.h"
#include "internals/cyphers/block.h"

#include "internals/cyphers/enigma.h"

#include "internals/cyphers/aes.h"

#include "internals/cyphers/scytale.h"
#include "internals/cyphers/transposition.h"
#include "internals/cyphers/fleissner.h"
#include "internals/cyphers/bifid.h"
#include "internals/cyphers/stream.h"
#include "internals/cyphers/stattests.h"
#include "internals/cyphers/stream.h"

#include "internals/enhancements/lith/lithuanian.h"

#define FLAG_COUNT 4

extern volatile char killswitch;

typedef struct
{
    char flags[FLAG_COUNT];
    const char *wordlist;
    unsigned char minLength;
    unsigned char maxLength;
    const char *specialRegex;

    char decrypt;
    char decypher;

    char transposition;
    char scytale;
    char fleissner;
    char bifid;
    char stream;

    char affineCaesar;
    char hill;
    char vigenere;
    char enigma;
    char feistel;
    char aes;
    char block;
    int stat;

    char enhancedBrute;

    char brute;
    const char *frag;
    const char *alph;

    char banner;

    const char *encText;
    const char *out;
} Arguments;