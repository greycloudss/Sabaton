#include "feistel.h"

/*
(m|k)^((k//16)&m)
(m&k)|((k%16)^m)
(m|k)^((k//16)&m)
(m&k)^((k%16)|m)

    int selection = 0;

    printf("Function 0:     (m|k)^((k//16)&m)\n");
    printf("Function 1:     (m&k)|((k\%16)^m) \n");
    printf("Function 2:     (m|k)^((k/16)&m)\n");
    printf("Function 3:     (m&k)^((k\%16)|m) \n");

    printf("Rare occasion - select your function: ");
    scanf("%[0-4]c", &selection);

    selection += 48


*/

int selFunc(char flag, uint8_t m, uint8_t k) {
    switch (flag) {
        case 0:
            return (uint8_t)((m | k) ^ ((k / 16) & m));

        case 1: 
            return (uint8_t)((m & k) | ((k % 16) ^ m));

        case 2:
            return (uint8_t)((m | k) ^ ((k / 16) & m));

        default:
            return (uint8_t)((m & k) ^ ((k % 16) | m));
    }
}


const char* pieceFesitel(const char* alph, const char* encText, int* keys, size_t n) {


    return "";
}


const char* partialFeistel(const char* alph, const char* encText, int* frag, size_t n) {
    return "";
}

const char* bruteFeistel(const char* alph, const char* encText) {
    return "";
} 

const char* feistelEntry(const char* alph, const char* encText, const char* frag){


    if (!frag || !*frag) return bruteFeistel(alph, encText);

    int n = 0;
    int* keys = parse_frag_array(frag, &n);
    if (!keys) return bruteFeistel(alph, encText);

    free(keys);
    return "";
}


