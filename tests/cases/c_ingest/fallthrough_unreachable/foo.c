/* Falling off the end of a non-main int-returning function is UB
   per C99 §6.9.1/12. The ingest must represent it explicitly so an
   analysis can flag the path — not silently synthesize `return 0`.
   This test pins the resulting Program shape. */
int foo(int x)
{
    if (x > 0) {
        return x + 1;
    }
    /* fall-through: UB */
}
