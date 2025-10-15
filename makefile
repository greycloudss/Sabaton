make:
	gcc ./main.c ./internals/hash.c ./internals/cyphers/affineCaesar.c ./internals/cyphers/enigma.c ./internals/cyphers/aes.c ./internals/cyphers/feistel.c ./internals/cyphers/block.c ./internals/cyphers/hill.c ./internals/cyphers/scytale.c  ./internals/cyphers/transposition.c ./internals/cyphers/vigenere.c ./internals/hashes/crc32.c ./internals/hashes/murmur3.c ./internals/hashes/sha1.c ./internals/hashes/sha256.c ./internals/hashes/xxhash32.c ./internals/lithuanian.c -lm -o sabaton
clean:
	rm sabaton