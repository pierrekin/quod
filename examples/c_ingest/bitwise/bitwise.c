/* Bitwise operators: shifts (<<, >>), xor (^), one's-complement (~),
   and logical-not (!). The C ingester lifts these to quod's core
   BinOp / CUnary nodes; the c-family lowering rewrites unary ! and ~
   to their BinOp identities (eq-zero and xor-with-minus-one). */

#include <stdio.h>

int low_bits(int x, int n) {
    /* keep the low n bits — mask via `((1 << n) - 1)`. */
    return x & ((1 << n) - 1);
}

int swap_nibbles(int x) {
    /* swap the low-byte nibbles of x: ((x & 0xF) << 4) | ((x >> 4) & 0xF). */
    return ((x & 15) << 4) | ((x >> 4) & 15);
}

int xor_round_trip(int x, int k) {
    /* xor is self-inverse: (x ^ k) ^ k == x. */
    return (x ^ k) ^ k;
}

int complement(int x) {
    /* ~x == -x - 1 over two's-complement ints. */
    return ~x;
}

int is_zero(int x) {
    /* logical-not: !x == 1 when x == 0, else 0. */
    return !x;
}

int main(void) {
    printf("low_bits(0xFF, 4)     = %d\n", low_bits(255, 4));
    printf("swap_nibbles(0x12)    = %d\n", swap_nibbles(18));
    printf("xor_round_trip(7, 13) = %d\n", xor_round_trip(7, 13));
    printf("complement(5)         = %d\n", complement(5));
    printf("is_zero(0)            = %d\n", is_zero(0));
    printf("is_zero(42)           = %d\n", is_zero(42));
    return 0;
}
