ifeq ($(OS),Windows_NT)
	CLEAN = del /F /Q sabaton.exe 2>nul || true
	OUT = sabaton.exe
else
	CLEAN = rm -f sabaton
	OUT = sabaton
endif

all:
	gcc ./main.c ./internals/hash.c \
	    ./internals/cyphers/affineCaesar.c ./internals/cyphers/enigma.c \
	    ./internals/cyphers/aes.c ./internals/cyphers/feistel.c \
	    ./internals/cyphers/block.c ./internals/cyphers/hill.c \
	    ./internals/cyphers/scytale.c ./internals/cyphers/transposition.c \
	    ./internals/cyphers/vigenere.c \
	    ./internals/hashes/crc32.c ./internals/hashes/murmur3.c \
	    ./internals/hashes/sha1.c ./internals/hashes/sha256.c \
	    ./internals/hashes/xxhash32.c ./internals/lithuanian.c \
	    ./internals/cyphers/bifid.c ./internals/cyphers/fleissner.c \
		./internals/cyphers/stream.c ./internals/cyphers/stattests.c \
	     -lm -o $(OUT)

clean:
	$(CLEAN)
