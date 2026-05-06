/* `do { body } while (cond);` — post-test loop. The body runs at
   least once, regardless of cond. Inside body, `continue` jumps to
   the cond check (matches C semantics). */

#include <stdio.h>

int count_down(int n) {
    /* Always runs once even if n <= 0. */
    int i = n;
    int iters = 0;
    do {
        iters = iters + 1;
        i = i - 1;
    } while (i > 0);
    return iters;
}

int sum_until(int n) {
    /* `continue` skips the accumulation step but the cond is
       still re-checked. */
    int i = 0;
    int total = 0;
    do {
        i = i + 1;
        if (i % 3 == 0) {
            continue;
        }
        total = total + i;
    } while (i < n);
    return total;
}

int main(void) {
    printf("count_down(5)   = %d\n", count_down(5));
    printf("count_down(-3)  = %d\n", count_down(-3));
    printf("sum_until(10)   = %d\n", sum_until(10));
    return 0;
}
