ifeq ($(OS),Windows_NT)
	CLEAN = del /F /Q sabaton.exe 2>nul || true & rmdir /S /Q build 2>nul || true
	OUT = sabaton.exe
	MKDIR = mkdir
else
	CLEAN = rm -f sabaton && rm -rf build
	OUT = sabaton
	MKDIR = mkdir -p
endif

BUILD = build

CC   = gcc
NVCC = nvcc

CUDA ?= 0

CFLAGS_COMMON = -O3 -march=native -mtune=native -ffast-math -funroll-loops \
                -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables

CFLAGS    = $(CFLAGS_COMMON)
NVCCFLAGS = -O3

ifeq ($(CUDA),1)
    CFLAGS    += -DUSE_CUDA
    NVCCFLAGS += -DUSE_CUDA
endif

C_SOURCES = \
	./main.c \
	./internals/hash.c \
	./internals/cyphers/affineCaesar.c ./internals/cyphers/enigma.c \
	./internals/cyphers/aes.c ./internals/cyphers/feistel.c \
	./internals/cyphers/block.c ./internals/cyphers/hill.c \
	./internals/cyphers/scytale.c ./internals/cyphers/transposition.c \
	./internals/cyphers/vigenere.c \
	./internals/hashes/crc32.c ./internals/hashes/murmur3.c \
	./internals/hashes/sha1.c ./internals/hashes/sha256.c \
	./internals/hashes/xxhash32.c ./internals/enhancements/lith/lithuanian.c \
	./internals/cyphers/bifid.c ./internals/cyphers/fleissner.c \
	./internals/cyphers/stream.c ./internals/cyphers/stattests.c

C_OBJECTS  = $(patsubst ./%.c,$(BUILD)/%.o,$(C_SOURCES))

ifeq ($(CUDA),1)
    CU_SOURCES = \
        ./internals/enhancements/cuda/feistel.cu \
        ./internals/enhancements/cuda/entry.cu
    CU_OBJECTS = $(patsubst ./%.cu,$(BUILD)/%.o,$(CU_SOURCES))
    OBJECTS    = $(C_OBJECTS) $(CU_OBJECTS)
else
    OBJECTS    = $(C_OBJECTS)
endif

all: $(OUT)

$(BUILD):
	$(MKDIR) $(BUILD)
	$(MKDIR) $(BUILD)/internals
	$(MKDIR) $(BUILD)/internals/cyphers
	$(MKDIR) $(BUILD)/internals/hashes
	$(MKDIR) $(BUILD)/internals/enhancements
	$(MKDIR) $(BUILD)/internals/enhancements/lith
	$(MKDIR) $(BUILD)/internals/enhancements/cuda

$(BUILD)/%.o: ./%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

ifeq ($(CUDA),1)
$(BUILD)/%.o: ./%.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

ifeq ($(CUDA),1)
$(OUT): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -lm -o $(OUT)
else
$(OUT): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -lm -o $(OUT)
endif

clean:
	$(CLEAN)
