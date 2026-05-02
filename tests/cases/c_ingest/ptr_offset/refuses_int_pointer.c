/* int* pointer arithmetic should refuse — quod.ptr_offset is byte-stride,
   not element-stride, so silently emitting it for int* would step by 1 byte
   instead of 4. The ingester rejects with a hint to cast to (char*).

   We get an int* into scope via an extern that returns one — function params
   can't be pointer-typed in v1, so this is the cleanest path. */

extern int *get_buf(void);
extern int read_it(int *p);

int main(void) {
    int *p = get_buf();
    return read_it(p + 1);
}
