/* Uninitialized locals (`int x;`). Layer A preserves the
   uninitialized declarator; layer B lifts to `Let(x, type, init=None)`.
   The validator's definite-init analysis refuses any program where a
   read of x can be reached before a definite write — preserving C's
   undefined-behaviour stance without silently zero-initializing. */

#include <stdio.h>

int branched_init(int n) {
    int x;
    if (n > 0) {
        x = n;
    } else {
        x = -n;
    }
    /* x is definitely written on every path before this read. */
    return x;
}

int sequential_init(int n) {
    int total;
    int i = 0;
    int sum = 0;
    while (i < n) {
        sum = sum + i;
        i = i + 1;
    }
    total = sum;  /* defines total */
    return total;
}

int main(void) {
    printf("branched_init(5)    = %d\n", branched_init(5));
    printf("branched_init(-3)   = %d\n", branched_init(-3));
    printf("sequential_init(5)  = %d\n", sequential_init(5));
    return 0;
}
