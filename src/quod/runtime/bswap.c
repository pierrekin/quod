// Byte-swap intrinsics exposed to quod via core.num.json (linkage.runtime).
// __builtin_bswap{16,32,64} is provided by both clang and gcc; with -O2
// each wrapper inlines to a single machine `bswap` instruction.

#include <stdint.h>

uint16_t quod_bswap_u16(uint16_t x) { return __builtin_bswap16(x); }
uint32_t quod_bswap_u32(uint32_t x) { return __builtin_bswap32(x); }
uint64_t quod_bswap_u64(uint64_t x) { return __builtin_bswap64(x); }
