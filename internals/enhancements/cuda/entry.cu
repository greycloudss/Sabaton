#ifdef USE_CUDA
#include "entry.h"
#include <string.h>
#include <stdio.h>


extern char g_funcFlag;
// void printASCII(void);

void entryCudaEnhancement(Arguments* args) {
    if (!args || !args->decypher)
        return;

    // if (args->banner)
    //     printASCII();

    if (args->feistel) {
        int sel;
        printf("Select GPU brute-force function mode (check in -h):\n");
        if (scanf(" %1d", &sel) != 1) sel = 0;
        if (sel < 0) sel = 0;
        if (sel > 4) sel = 4;
        g_funcFlag = (char)sel;

        const char* res = feistelBrute(args->alph, args->encText, args->frag);
        args->out = res;
        return;
    }

    if (args->stream) {
        const char* res = streamBruteCuda(args->alph, args->encText, args->frag);
        args->out = res;
        return;
    }

    if (args->rsa) {
        const char* res = rsaBruteCuda(args->alph, args->encText, args->frag);
        args->out = res;
        return;
    }

    if (args->rabin) {
        const char* res = rabinBruteCuda(args->alph, args->encText, args->frag);
        args->out = res;
        return;
    }

    if (args->merkle) {
        const char* res = merkleBruteCuda(args->alph, args->encText, args->frag);
        args->out = res;
        return;
    }

    // if (args->block) {
    //     if (args->brute || !args->frag) {
    //         const char* res = blockEntry(args->encText, NULL, 0);
    //         args->out = res;
    //         return;
    //     } else {
    //         char flag = 0;
    //         char* keys = NULL;
    //         feistel_extract(args->frag, &flag, &keys);
    //         const char* res = blockEntry(args->encText, keys, flag);
    //         if (keys) free(keys);
    //         args->out = res;
    //         return;
    //     }
    // }



    // if (args->bifid) {
    //     const char* frag = args->brute ? NULL : args->frag;
    //     const char* res = bifidEntry(args->alph, args->encText, frag);
    //     args->out = res;
    //     return;
    // }

    // if (args->stream) {
    //     const char* frag = args->brute ? NULL : args->frag;
    //     const char* res = streamEntry(args->alph, args->encText, frag);
    //     args->out = res;
    //     return;
    // }

    // if (args->stat) {
    //     const char* res = statEntry(args->alph, args->encText);
    //     args->out = res;
    //     return;
    // }

    // if (args->transposition) {
    //     const char* frag = args->brute ? NULL : args->frag;
    //     const char* res = transpositionEntry(args->alph, args->encText, frag);
    //     args->out = res;
    //     return;
    // }

    // if (args->fleissner) {
    //     const char* frag = args->brute ? NULL : args->frag;
    //     const char* res = fleissnerEntry(args->alph, args->encText, frag);
    //     args->out = res;
    //     return;
    // }

    // if (args->hill) {
    //     const char* frag = args->brute ? NULL : args->frag;
    //     const char* res = hillEntry(args->alph, args->encText, frag);
    //     args->out = res;
    //     return;
    // }

    // if (args->affineCaesar) {
    //     const char* frag = args->brute ? NULL : args->frag;
    //     const char* res = affineCaesarEntry(args->alph, args->encText, frag);
    //     args->out = res;
    //     return;
    // }

    // if (args->enigma) {
    //     const char* frag = args->brute ? NULL : args->frag;
    //     const char* res = enigmaEntry(args->alph, args->encText, frag);
    //     args->out = res;
    //     return;
    // }

    // if (args->aes) {
    //     const char* frag = args->brute ? NULL : args->frag;
    //     const char* res = aesEntry(args->alph, args->encText, frag);
    //     args->out = res;
    //     return;
    // }

    // if (args->vigenere) {
    //     const char* frag = args->brute ? NULL : args->frag;
    //     const char* res = vigenereEntry(args->alph, args->encText, frag);
    //     args->out = res;
    //     return;
    // }
}

#endif
