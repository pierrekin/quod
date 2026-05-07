/* `long double` is implementation-defined and x87-only on the
   Linux target; quod refuses it at every type-resolution site. */

long double f(long double x) {
    return x;
}

int main(void) {
    return 0;
}
