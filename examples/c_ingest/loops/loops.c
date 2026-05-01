/* While loops with locals + assignment. Note that v1 doesn't ingest `for`,
   so a counter loop is written as `int i = 0; while (i < n) { ...; i = i + 1; }`. */

#include <stdio.h>

int sum_to(int n) {
    int total = 0;
    int i = 1;
    while (i <= n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}

int factorial(int n) {
    int result = 1;
    int i = 2;
    while (i <= n) {
        result = result * i;
        i = i + 1;
    }
    return result;
}

int main(void) {
    printf("sum_to(10)    = %d\n", sum_to(10));
    printf("factorial(6)  = %d\n", factorial(6));
    return 0;
}
