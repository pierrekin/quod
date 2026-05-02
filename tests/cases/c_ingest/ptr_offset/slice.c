/* Pointer arithmetic: `p + n` and `&buf[k]` patterns over char* / char[].
   Both should ingest as quod.ptr_offset over the char-pointer base.

   v1 restrictions:
     - Pointee must be `char` (byte stride matches quod's GEP).
     - Offset must be an integer literal (no sext yet for variable offsets).
*/

#include <stdio.h>

int main(void) {
    char *full = "hello, world";
    printf("plus  = %s\n", full + 7);
    printf("subscr= %s\n", &full[7]);
    return 0;
}
