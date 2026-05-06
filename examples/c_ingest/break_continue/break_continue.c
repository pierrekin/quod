/* `break` exits the innermost loop; `continue` skips to the next
   iteration. Inside a `for` loop, `continue` jumps to the inc step
   (not the cond) — the c-family lowering pre-rewrites Continue to
   `inc; continue` to preserve that semantic. */

#include <stdio.h>

int find_first_negative(int n) {
    /* Scan 0..n-1; break at the first negative residue. */
    int i = 0;
    while (i < n) {
        if (i - 5 < 0) {
            return i - 5;
        }
        if (i >= 100) {
            break;  /* defensive cap */
        }
        i = i + 1;
    }
    return -999;
}

int sum_evens(int n) {
    /* `continue` inside a for-loop must run the inc — without the
       c-family rewrite we'd loop forever. */
    int total = 0;
    for (int i = 0; i < n; i = i + 1) {
        if (i % 2 != 0) {
            continue;
        }
        total = total + i;
    }
    return total;
}

int main(void) {
    printf("find_first_negative(20) = %d\n", find_first_negative(20));
    printf("sum_evens(10)           = %d\n", sum_evens(10));
    return 0;
}
