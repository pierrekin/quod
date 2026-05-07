/* Pre/post-increment and -decrement: `i++`, `++i`, `i--`, `--i`.
   Each lifts to a layer-A c.increment_stmt (preserving the source
   operator and pre/post position) and desugars to
   `Assign(i, BinOp("add"|"sub", LocalRef(i), IntLit(1)))` on layer B.
   Pre and post are observably identical in statement position; the
   distinction is preserved at layer A for source fidelity. */

#include <stdio.h>

int sum_to(int n) {
    /* `i++;` as a bare while-body statement. */
    int total = 0;
    int i = 0;
    while (i < n) {
        total = total + i;
        i++;
    }
    return total;
}

int countdown(int n) {
    /* `--i;` as a bare while-body statement. */
    int steps = 0;
    int i = n;
    while (i > 0) {
        --i;
        steps = steps + 1;
    }
    return steps;
}

int loop_post(int n) {
    /* `i++` in for-loop inc position. */
    int total = 0;
    for (int i = 0; i < n; i++) {
        total = total + i;
    }
    return total;
}

int loop_pre(int n) {
    /* `++i` in for-loop inc position — observably identical to i++ here. */
    int total = 0;
    for (int i = 0; i < n; ++i) {
        total = total + i;
    }
    return total;
}

int loop_dec(int n) {
    /* `--i` in for-loop inc position. */
    int total = 0;
    for (int i = n; i > 0; --i) {
        total = total + 1;
    }
    return total;
}

int main(void) {
    printf("sum_to(10)    = %d\n", sum_to(10));
    printf("countdown(10) = %d\n", countdown(10));
    printf("loop_post(10) = %d\n", loop_post(10));
    printf("loop_pre(10)  = %d\n", loop_pre(10));
    printf("loop_dec(10)  = %d\n", loop_dec(10));
    return 0;
}
