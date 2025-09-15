#include "main.h"
#include <stdio.h>

volatile char killswitch = 0;


void printHelp() {
    printf("Usage:\n");
    printf("  <prog> -decypher -affineCaesar -alph <alphabet> [-frag <fragment> | -brute]\n");
    printf("\n");
    printf("-decypher           Enable cipher-decoding mode.\n");
    printf("-affineCaesar       Select the affine Caesar cipher module.\n");
    printf("-alph <alphabet>    Alphabet string to operate on; its length m defines modulo m.\n");
    printf("                    Characters not in this string pass through unchanged.\n");
    printf("-frag <fragment>    Known plaintext snippet (e.g., prefix). Uses it to solve keys (a,b)\n");
    printf("                    directly and decrypt once; fastest if the snippet is correct.\n");
    printf("-brute              Try all valid keys (a coprime with m; b in [0..m-1]) and output each\n");
    printf("                    candidate plaintext with its keys.\n");
    printf("\n");
    printf("Notes:\n");
    printf("  • Provide -alph with the exact ordering you expect (case/diacritics included).\n");
    printf("  • Prefer -frag when you know a snippet; use -brute when you don't.\n");
    printf("  • If both -frag and -brute are given, the tool will attempt -frag first and may fall back to -brute.\n");
    printf("\n");
    printf("Examples:\n");
    printf("  <prog> -decypher -affineCaesar -alph \"ABCDEFGHIJKLMNOPQRSTUVWXYZ\" -frag \"THE\"\n");
    printf("  <prog> -decypher -affineCaesar -alph \"AĄBCČDEĘĖFGHIY...Ž\" -brute\n");
}

void parseArgs(Arguments* args, const int argv, const char** argc) {
    memset(args->flags, 0, sizeof(args->flags));
    args->decrypt = 0;
    args->decypher = 0;
    args->affineCaesar = 0;
    args->brute = 0;
    args->frag = NULL;
    args->alph = NULL;
    args->wordlist = NULL;
    args->encText = NULL;
    args->out = NULL;

    for (int i = 1; i < argv; ++i) {
        const char* a = argc[i];

        if (strcmp(a, "-h") == 0) {
            printHelp();
            return;
        }

        if (strcmp(a, "-decrypt") == 0) {
            args->decrypt = 1;
            continue;
        }

        if (strcmp(a, "-decypher") == 0) {
            args->decypher = 1;
            continue;
        }

        if (args->decrypt) {
            if (strcmp(a, "-w") == 0) {
                if (i + 1 < argv) { args->flags[0] = 1; args->wordlist = argc[++i]; }
                continue;
            }
            if (strcmp(a, "-ml") == 0) {
                if (i + 1 < argv) { args->flags[1] = 1; args->minLength = (unsigned char)stoi(argc[++i]); }
                continue;
            }
            if (strcmp(a, "-xl") == 0) {
                if (i + 1 < argv) { args->flags[2] = 1; args->maxLength = (unsigned char)stoi(argc[++i]); }
                continue;
            }
            if (strcmp(a, "-s") == 0) {
                if (i + 1 < argv) { args->flags[3] = 1; args->specialRegex = argc[++i]; }
                continue;
            }
        }

        if (args->decypher) {
            if (strcmp(a, "-affineCaesar") == 0) {
                args->affineCaesar = 1;
                continue;
            }
            if (strcmp(a, "-brute") == 0) {
                args->brute = 1;
                continue;
            }
            if (strcmp(a, "-frag") == 0) {
                if (i + 1 < argv) { args->frag = argc[++i]; }
                continue;
            }
            if (strcmp(a, "-alph") == 0) {
                if (i + 1 < argv) { args->alph = argc[++i]; }
                continue;
            }
        }

        if (a[0] != '-' && !args->encText) {
            args->encText = a;
            continue;
        }
    }
}

void decypher(Arguments* args) {
    if (!args || !args->decypher) return;

    if (args->affineCaesar) {
        const char* frag = args->brute ? NULL : args->frag;
        const char* res = affineCaesarEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }
}


int main(int argv, const char** argc) {
    Arguments args = { .minLength = 0, .maxLength = 244, .specialRegex = "[!\"#$%&'()*+,-./:;<=>?@[\\\\\\]^_`{|}~]" };
    parseArgs(&args, argv, argc);
    decypher(&args);
    return 0;
}