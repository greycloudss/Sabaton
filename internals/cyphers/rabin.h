#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../util/string.h"
#include "../../util/number.h"
#include "../../util/fragmentation.h"
#include "../../util/bigint.h"


const char* rabinEntry(const char* alph, const char* encText, const char* frag);