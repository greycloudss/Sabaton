#pragma once

#define FLAG_COUNT 4

typedef struct {
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
    char rabin;
    char zkp;
    char affineCaesar;
    char hill;
    char vigenere;
    char enigma;
    char feistel;
    char elgamal;
    char aes;
    char block;
    char stat;
    char graham;
    char merkle;
    char ellipticCurve;
    char rsa;

    char enhancedBrute;
    char gpu;

    char brute;
    const char *frag;
    const char *alph;

    char banner;

    const char *encText;
    const char *out;
} Arguments;
