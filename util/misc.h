#pragma once

//here i will house stuff like id in alphabet which are more for cryptography than for a specific thing

static int indexInAlphabet(const char* alph, unsigned char c) {
    for (int i = 0; alph[i]; ++i) {
        if ((unsigned char)alph[i] == c) return i;
    }
    return -1;
}
