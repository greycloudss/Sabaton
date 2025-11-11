#ifndef MERKLE_H
#define MERKLE_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../util/string.h"
#include "../../util/number.h"
#include "knapsack.h"



const char* merkleEntry(const char* alph, const char* encText, const char* frag);


#endif