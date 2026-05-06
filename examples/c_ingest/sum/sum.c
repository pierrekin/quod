/* Worked example for the staged-lift C-ingest design (see
   .scratch/c-ingest/00-overview.md). The fully-populated `for`
   triggers the layer-A → layer-B path: layer A preserves the C
   subtree as `c_unit` / `c.fn` / `c.for` / …; layer B transcribes
   it to core quod plus `CStyleFor` (`c.for_general`).

   `lower.py` refuses programs containing `c.for_general` until the
   c-family lowering pass (lower/c_family.py, step 5) lands. */

int sum(int n) {
    int s = 0;
    for (int i = 0; i < n; i = i + 1) {
        s = s + i;
    }
    return s;
}
