/* Compound assignments: `x += y`, `x -= y`, `x *= y`, `x /= y`,
   `x %= y`, `x &= y`, `x |= y`, `x ^= y`, `x <<= y`, `x >>= y`.
   Each lifts to a layer-A c.compound_assign and desugars to
   `Assign(x, BinOp(op_translated, LocalRef(x), y'))` on layer B. */

#include <stdio.h>

int sum_to(int n) {
    int total = 0;
    int i = 1;
    while (i <= n) {
        total += i;       /* +=  */
        i += 1;           /* +=  */
    }
    return total;
}

int reduce(int n) {
    /* exercise -=, *=, /=, %= in one body. */
    int x = n;
    x *= 3;
    x -= 4;
    x /= 2;
    x %= 100;
    return x;
}

int bit_ops(int x, int n) {
    /* exercise &=, |=, ^=, <<=, >>=. */
    int r = x;
    r &= 0xFF;        /* low byte */
    r |= 0x10;        /* set bit 4 */
    r ^= 0x05;        /* flip bits 0 and 2 */
    r <<= n;          /* left-shift by n */
    r >>= 1;          /* arithmetic right-shift by 1 */
    return r;
}

int main(void) {
    printf("sum_to(10)         = %d\n", sum_to(10));
    printf("reduce(50)         = %d\n", reduce(50));
    printf("bit_ops(0x1234, 2) = %d\n", bit_ops(0x1234, 2));
    return 0;
}
