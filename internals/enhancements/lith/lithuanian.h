#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../../util/string.h"
#include "../../../util/number.h"
#include "../../../util/misc.h"

// a smidge of phonetic recognition


const char* getExtension(const char* filename);

const char* recognEntry(const char* bruteFile);