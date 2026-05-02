/* quod_arena: tiny bump allocator with chunk extension.
 *
 * Pointers handed out by quod_arena_alloc remain valid until the matching
 * quod_arena_drop. The allocator never relocates: when a bump runs out of
 * room, we add a fresh chunk to a singly-linked list and serve from there.
 *
 * Public surface (the names matter — these are what quod externs declare):
 *   quod_arena_new   (capacity)         -> opaque arena handle (i8*)
 *   quod_arena_alloc (arena, n_bytes)   -> pointer into arena (i8*) or NULL on OOM
 *   quod_arena_drop  (arena)            -> 0
 *   quod_arena_used  (arena)            -> total bytes currently allocated
 *
 * Everything is i64 for sizes — quod has no size_t and we don't want any
 * width games at the FFI boundary. Alignment: 16 bytes, generous enough
 * for anything quod will hand us in v1.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QUOD_ARENA_ALIGN 16

typedef struct Chunk {
    struct Chunk *prev;     /* older chunk in the chain, or NULL */
    int64_t        cap;     /* total bytes in `data` */
    int64_t        cur;     /* bytes used in `data` */
    /* `data` begins immediately after the header (we allocate header + cap
     * in a single malloc). */
} Chunk;

typedef struct Arena {
    Chunk  *head;           /* most recent chunk; allocations go here first */
    int64_t default_cap;    /* size of the next chunk we grow into */
    int64_t total_used;     /* sum of `cur` across all chunks (introspection) */
} Arena;

static int64_t round_up(int64_t n, int64_t align) {
    return (n + (align - 1)) & ~(align - 1);
}

static Chunk *new_chunk(int64_t capacity) {
    /* +sizeof(Chunk) for the header; round capacity up so `data` is aligned. */
    int64_t cap = round_up(capacity, QUOD_ARENA_ALIGN);
    if (cap < 1) cap = QUOD_ARENA_ALIGN;
    Chunk *c = (Chunk *)malloc(sizeof(Chunk) + (size_t)cap);
    if (!c) return NULL;
    c->prev = NULL;
    c->cap = cap;
    c->cur = 0;
    return c;
}

static char *chunk_data(Chunk *c) {
    return (char *)(c + 1);
}

void *quod_arena_new(int64_t initial_capacity) {
    Arena *a = (Arena *)malloc(sizeof(Arena));
    if (!a) return NULL;
    int64_t cap = initial_capacity > 0 ? initial_capacity : 4096;
    Chunk *c = new_chunk(cap);
    if (!c) { free(a); return NULL; }
    a->head = c;
    a->default_cap = cap;
    a->total_used = 0;
    return a;
}

void *quod_arena_alloc(void *arena, int64_t n_bytes) {
    if (!arena || n_bytes < 0) return NULL;
    Arena *a = (Arena *)arena;
    int64_t need = round_up(n_bytes < 1 ? 1 : n_bytes, QUOD_ARENA_ALIGN);

    Chunk *c = a->head;
    if (c->cur + need > c->cap) {
        /* Single allocation exceeds the default chunk size? grow to fit. */
        int64_t cap = a->default_cap;
        if (need > cap) cap = need;
        Chunk *nc = new_chunk(cap);
        if (!nc) return NULL;
        nc->prev = c;
        a->head = nc;
        c = nc;
    }
    void *p = chunk_data(c) + c->cur;
    c->cur += need;
    a->total_used += need;
    /* Zero the slab — quod's allocator-of-record is meant to feel like
     * calloc by default. Cheap relative to real work in v1. */
    memset(p, 0, (size_t)need);
    return p;
}

int64_t quod_arena_drop(void *arena) {
    if (!arena) return 0;
    Arena *a = (Arena *)arena;
    Chunk *c = a->head;
    while (c) {
        Chunk *prev = c->prev;
        free(c);
        c = prev;
    }
    free(a);
    return 0;
}

int64_t quod_arena_used(void *arena) {
    if (!arena) return 0;
    return ((Arena *)arena)->total_used;
}
