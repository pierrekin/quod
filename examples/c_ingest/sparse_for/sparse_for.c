/* Sparse for-loops: any of init, cond, inc may be absent. The
   c-family lowering rule rewrites `for (init; cond; inc) body` to
   `init; while (cond) { body; inc; }`, with an absent cond becoming
   `while (true)`. */

#include <stdio.h>

int sum_no_init(int n) {
    int i = 0, total = 0;
    /* init slot empty: i is set up by the caller's preceding decls. */
    for (; i < n; i = i + 1) {
        total = total + i;
    }
    return total;
}

int sum_no_inc(int n) {
    /* inc slot empty: increment lives inside the body. */
    int total = 0;
    for (int i = 0; i < n; ) {
        total = total + i;
        i = i + 1;
    }
    return total;
}

int sum_no_cond(int n) {
    /* cond slot empty: while(true) loop, body must break out — here we
       use an early `return` (since `break` lands in a later commit). */
    int total = 0;
    for (int i = 0; ; i = i + 1) {
        if (i >= n) { return total; }
        total = total + i;
    }
}

int main(void) {
    printf("sum_no_init(5)  = %d\n", sum_no_init(5));
    printf("sum_no_inc(5)   = %d\n", sum_no_inc(5));
    printf("sum_no_cond(5)  = %d\n", sum_no_cond(5));
    return 0;
}
