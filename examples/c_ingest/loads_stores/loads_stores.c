/* Loads, stores, and typed pointer arithmetic — the Stage A unlock
   for the C ingester. Lifts to:
       *p           ↔ Load(p, T)
       arr[k]       ↔ Load(PtrOffset(p, k * sizeof(T)), T)
       *p = v       ↔ Store(p, v)
       arr[k] = v   ↔ Store(PtrOffset(p, k * sizeof(T)), v)
       int *p + n   ↔ PtrOffset(p, n * sizeof(int))

   Local stack arrays / malloc / globals aren't ingested in this
   stage; callers are responsible for the storage these functions
   read and write. */

int deref(int *p) {
    return *p;
}

int index_at(int *p, int k) {
    return p[k];
}

void set(int *p, int v) {
    *p = v;
}

void set_at(int *p, int k, int v) {
    p[k] = v;
}

int read_after(int *p, int n) {
    return *(p + n);
}

int swap(int *p, int *q) {
    int tmp = *p;
    *p = *q;
    *q = tmp;
    return 0;
}
