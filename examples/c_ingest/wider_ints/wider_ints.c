/* Wider integer types: char/short/int/long/long_long, signed and
   unsigned, plus typedef'd standards (size_t, int64_t, uint8_t, ...).

   Each lifts to a layer-A `CNamedType(name=<source spelling>)` and a
   layer-B quod IntType chosen by clang's canonical TypeKind. The
   lift-check canonicalizes spellings via `_C_TYPE_NAME_TO_QUOD_KIND`.

   Mixed-width arithmetic emits explicit `Cast` nodes at layer B (no
   layer-A counterpart — they're clang-inserted promotions, not source
   syntax). Signedness drives BinOp dispatch: unsigned operands get
   `udiv`/`urem`/`ult`/`...`/`lshr` instead of the signed variants. */

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

/* Wider-than-int signed/unsigned arithmetic. */
int64_t add_i64(int64_t a, int64_t b) {
    return a + b;
}

uint64_t add_u64(uint64_t a, uint64_t b) {
    return a + b;
}

/* Unsigned division — must emit `udiv`, not `sdiv`. */
unsigned int udiv_test(unsigned int a, unsigned int b) {
    return a / b;
}

/* Unsigned right-shift — must emit `lshr`, not `ashr`. */
uint32_t lshr_test(uint32_t x, int n) {
    return x >> n;
}

/* size_t typedef — resolves to unsigned long on Linux LP64 (= u64). */
size_t sum_sz(size_t a, size_t b) {
    return a + b;
}

/* Implicit promotion: u8 + u8 → int. The result type is `int` per C99
   §6.3.1.8 (operands narrower than `int` are promoted to `int`).
   Layer B emits `Cast` nodes around each operand. */
int sum_u8(unsigned char a, unsigned char b) {
    return a + b;
}

/* Explicit cast: `(int64_t)x` lifts to a layer-A `CCast` node and a
   layer-B `Cast` (sext widening from i32 to i64). */
int64_t widen_to_i64(int x) {
    return (int64_t)x;
}

int main(void) {
    printf("add_i64(big, big)      = %lld\n", (long long)add_i64(2000000000LL, 2000000000LL));
    printf("add_u64(huge, huge)    = %llu\n", (unsigned long long)add_u64(0x8000000000000000ULL, 1ULL));
    printf("udiv_test(0x80000000U, 2U) = %u\n", udiv_test(0x80000000U, 2U));
    printf("lshr_test(0x80000000U, 1) = %u\n", lshr_test(0x80000000U, 1));
    printf("sum_sz(100, 200)       = %zu\n", sum_sz((size_t)100, (size_t)200));
    printf("sum_u8(200, 100)       = %d\n", sum_u8(200, 100));
    printf("widen_to_i64(-7)       = %lld\n", (long long)widen_to_i64(-7));
    return 0;
}
