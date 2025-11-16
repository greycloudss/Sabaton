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
#include "internals/enhancements/cuda/entry.h"

extern volatile char killswitch;