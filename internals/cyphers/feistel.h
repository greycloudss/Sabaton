#pragma once
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "../../util/string.h"
#include "../../util/number.h"
#include "../../util/misc.h"

#define MAX_FUNCS 5

const char* feistelEntry(const char* encText, const char* frag, char flag);