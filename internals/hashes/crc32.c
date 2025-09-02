#include "crc32.h"

u32 crc32_reflected(const unsigned char* data, size_t len) {
    u32 crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (u32) data[i];
        for (int k = 0; k < 8; ++k) {
            u32 mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}
