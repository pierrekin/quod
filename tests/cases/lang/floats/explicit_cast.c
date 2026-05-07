/* Explicit C casts via the ingester: `(int)x` lifts to a CCast at
   layer A and a Cast(target_type=I32Type) at layer B, which lowers to
   `llvm.fptosi.sat.i32.f64`. Truncation toward zero is the IEEE
   behavior for finite double → int (3.7 → 3, -2.1 → -2). */

#include <stdio.h>

int main(void) {
    printf("(int)3.7  = %d\n", (int)3.7);
    printf("(int)-2.1 = %d\n", (int)-2.1);
    printf("(int)0.9  = %d\n", (int)0.9);
    return 0;
}
