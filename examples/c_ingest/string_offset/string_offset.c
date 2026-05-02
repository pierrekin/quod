/* Pointer arithmetic over char*: walks a string by byte offset.
   Demonstrates two equivalent ways to slice — `p + n` and `&p[n]` — both
   ingest as quod.ptr_offset.

   v1 ingester restrictions:
     - The pointer must be char-typed (byte-stride matches quod's GEP).
     - Offsets must be integer literals (variable offsets need an
       i32 → i64 widening expression that quod doesn't yet expose).
*/

#include <stdio.h>

int main(void) {
    char *greeting = "hello, world!";
    printf("full   = %s\n", greeting);
    printf("p + 7  = %s\n",  greeting + 7);
    printf("&p[7]  = %s\n", &greeting[7]);
    printf("p + 12 = %s\n",  greeting + 12);
    return 0;
}
