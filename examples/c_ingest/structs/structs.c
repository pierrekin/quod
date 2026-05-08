/* Stage B unlock — struct definitions, by-value reads, struct-pointer
   reads, struct-pointer writes, aggregate initialisers. Lifts to:
       struct Foo {...};   ↔ StructDef(name='Foo', fields=...)
       (struct Foo){a, b}  ↔ StructInit(type='Foo', fields=[...])
       p.x                 ↔ FieldRead(value=p, name='x')
       p->x                ↔ LoadField(ptr=p, struct_type='Foo', name='x')
       p->x = v;           ↔ StoreField(ptr=p, struct_type='Foo',
                                         name='x', value=v)

   `p.x = v` (by-value field assign on a local struct) is refused —
   use the pointer form. Anonymous structs, unions, bit-fields are
   refused. */

struct Point { int x; int y; };

int sum_xy(struct Point p) {
    return p.x + p.y;
}

int sum_via_ptr(struct Point *p) {
    return p->x + p->y;
}

void set_x(struct Point *p, int v) {
    p->x = v;
}

void set_xy(struct Point *p, int x, int y) {
    p->x = x;
    p->y = y;
}

int build_and_sum(int a, int b) {
    struct Point p = {a, b};
    return p.x + p.y;
}

struct Span { int *data; int len; };

int span_sum(struct Span s) {
    int total = 0;
    int i = 0;
    while (i < s.len) {
        total = total + s.data[i];
        i = i + 1;
    }
    return total;
}
