# quod — module system, generics, parameterization

How quod modules expose parameters and how callers fill them. Two layers
compose: **wirables** (import-time-bound, declared at module scope) and
**type-level parameters** (construction-time-bound, declared on a type
or function). Read `LANGUAGE.md` for the surface syntax of imports and
generics; this document is the architecture behind it.

## What the system does

A module like `alloc.list` exposes a parameterized container
(`List<T>`) that needs an allocator to do anything. The allocator is
*not* a runtime value threaded through every call — it is part of the
type. Each `(T, allocator)` instantiation produces a fresh nominal
struct; allocator state is inlined where the optimizer can see it; the
claim system can reason per-allocator (`fresh`, `disjoint`, `aligned`
look very different for an arena vs a GPA).

The trade is more monomorphizations for less indirect-call overhead and
a denser claim surface. Type erasure is available as a deliberate
opt-in seam, never the default.

## The two parameterization layers

Two distinct binding sites:

- **Wirable** — declared at the top of a module
  (`wirable A: Allocator`). Bound at *import time* by the wirer. Acts
  like a type parameter in scope throughout the module, but does *not*
  appear as a slot in type signatures at use sites — once wired, it is
  gone.
- **Type-level parameter** — declared on a type or function (`struct
  List<T> { ... }`, `fn read<A: Allocator>(...)`). Bound at
  *construction or call time* by the user.

A library author chooses per-slot which kind to use. The decision
hinges on who picks the value and when:

- *The library's caller picks per-construction* → type-level (`<T>`).
- *The library's importer picks once for everything in this module* →
  wirable.

`wire` at the import site fills **wirables only**. Type-level
parameters remain user-bound at the construction site. This is what
makes the two layers compose cleanly: a wired-into module's exports
have a smaller type-arg surface than the same module unwired, and the
remaining type-arg slots are exactly what the user is supposed to
choose per-use.

Concretely:

```
module alloc.list {
  wirable A: Allocator                    // import-bound slot
  struct List<T> { ptr: *u8, len: i64, cap: i64, alloc: *A }
  //         ^-- type-level T; A is in scope from the wirable
  fn push<T>(self: *List<T>, x: T) -> Result<(), AllocError> { ... }
}
```

A wirer that says `import alloc.list wire A=Arena` sees:

```
List<i64>              // valid; A is already Arena
List<JsonValue>        // valid; A is already Arena
push(&xs, 42)          // calls List<i64, Arena>::push under the hood
```

No `<i64, Arena>` ever appears at the use site, because `A` was
bound *at import*, not *at construction*. The library author chose
that by writing `wirable A` instead of `struct List<T, A>`.

## Multiple wirables

A library can have multiple wirables and the importer fills each by
name:

```
module alloc.hashmap {
  wirable BucketAlloc: Allocator
  wirable ItemAlloc:   Allocator

  import alloc.list wire A=BucketAlloc       // List uses BucketAlloc
  struct HashMap<K, V> {
    buckets: List<Bucket<K, V>>,             // BucketAlloc-backed
    item_alloc: *ItemAlloc,                  // for owned values
  }
}
```

Each wirable is independently filled:

```
import alloc.hashmap wire BucketAlloc=Gpa, ItemAlloc=Arena
// Or use one allocator for both:
import alloc.hashmap wire BucketAlloc=Arena, ItemAlloc=Arena
```

The library author exposes the slots they care about distinguishing,
the user fills them per-policy. A library that genuinely only needs one
allocator declares one wirable; a library that benefits from
fine-grained control declares two.

## Wiring at import

Three import-site composers, orthogonal:

- `import X` — pull module `X`'s symbols into scope. Whole-module today;
  per-symbol later (see "Not yet" below).
- `import X wire P=Y` — bind module `X`'s wirable `P` to `Y`. Multiple
  bindings are comma-separated: `import X wire P=Y, Q=Z`.
- `import X as Z` — rename. The whole module is reachable as `Z.*`.

All three compose:

```
import alloc.list as ml wire A=mem.arena.Arena
```

Names never collide across distinct imports. If you want the same
module wired two different ways in the same program, rename:

```
import alloc.list as arena_list wire A=mem.arena.Arena
import alloc.list as gpa_list   wire A=mem.gpa.Gpa
// arena_list.List<T> and gpa_list.List<T> are distinct nominal types.
```

Without `as`, the second import is a duplicate-name error.

## Library pattern: forwarding through your own wirable

A library that doesn't want to commit to an allocator declares its own
matching wirable and forwards it at the inner import. Explicit
re-declaration, no implicit propagation:

```
module mylib {
  wirable A: Allocator
  import alloc.list wire A=A          // forward our own wirable
  fn process(xs: alloc.list.List<i64>) { ... }
}
```

`mylib` is now itself a parameterized module. It cannot be used in a
program without wiring `A`. The application is the only place where
everything resolves:

```
program {
  import mem.arena.Arena
  import mem.gpa.Gpa
  import mylib as ml  wire A=Arena
  import mylib as ml2 wire A=Gpa
  fn main() {
    ml.process(...)
    ml2.process(...)
  }
}
```

The cost is paid by library authors who pass through 3+ wirables. In
practice, libraries with that many free wirables are doing too much and
should split. The benefit: every module's wirable surface is fully
self-describing — no need to walk imports to know what's free.

A module with unwired `wirable`s is *not* directly usable in a program —
it must be wired at import, or its wirables must be propagated through
another module that itself ends up wired. Programs are the bottom of
the stack and have no remaining free wirables by definition.

## Monomorphization

The compiler walks every concrete instantiation and emits one fresh
nominal struct + bound bodies per `(template, args)` tuple,
deduplicated. Mangled names so `List<JsonValue, Arena>::push` and
`List<u8, Arena>::push` co-exist. Generic *impls*
(`impl<T> Trait for Box<T>`) are unified against the concrete
instantiation and instantiated on demand.

`SelfType` inside an `impl`'s methods (params, return type, *and* body
positions like `let x: Self = ...`) is rewritten to the impl's
`for_type` at construction. The lowerer never sees `Self`.

Bound enforcement runs at the binding site: if `T: Foo` and there is no
`impl Foo for <arg>` in scope, the instantiation errors with
`"parameter 'T' is bound by 'Foo', but no 'impl Foo for <arg>' is in
scope"` — even if the body never dispatches the bound trait. This
catches mistakes early instead of at first dispatch.

## Erasure seam

When you genuinely need a function to accept a container regardless of
its allocator (debug printer, test helper, dynamic plugin boundary),
there is an explicit erasure seam: a `dyn Alloc` fat-pointer form
analogous to a value-allocator. The seam is **named**, not free. Most
APIs avoid it; it shows up at boundaries where the cost is acceptable.

Concrete shape: TBD. Likely `&dyn Alloc` and a special
`dyn`-instantiation that boxes the allocator. The point is that
erasure becomes a deliberate API choice, not the default.

## Slice / owned remains a two-type split

`Slice<T>` is allocator-free by construction (it is a borrow);
`List<T>` (with a wired allocator) is the owning, parameterized form.
Same for `Str` (slice) vs `StrBuf` (owned). Don't try to unify these via
`A = ()` — different roles, different APIs.

## Worked examples

### Embedded SRAM with a custom allocator

```
// mem/fixed.qs
module mem.fixed {
  import mem.{Allocator, AllocError}
  import core.result.{Result, Ok, Err}

  struct FixedBuffer { buf: *u8, cap: i64, cursor: i64 }

  fn wrap(buf: *u8, cap: i64) -> FixedBuffer {
    FixedBuffer { buf: buf, cap: cap, cursor: 0 }
  }

  impl Allocator for FixedBuffer {
    fn alloc(self: *FixedBuffer, n: i64) -> Result<*u8, AllocError> {
      let aligned = (self.cursor + 7) & ~7    // 8-byte align
      if aligned + n > self.cap {
        return Err(AllocError::OutOfMemory)
      }
      let p = self.buf + aligned
      self.cursor = aligned + n
      Ok(p)
    }
    fn free(self: *FixedBuffer, _p: *u8) { /* bump-only; no individual free */ }
  }
}

// app/embedded_demo.qs
import mem.fixed.FixedBuffer
import alloc.list  wire A = FixedBuffer
import alloc.str   wire A = FixedBuffer

const SRAM_BASE: *u8 = 0x2000_0000 as *u8
const SRAM_SIZE: i64 = 4096

fn main() -> i32 {
  let mut sram   = FixedBuffer::wrap(SRAM_BASE, SRAM_SIZE)
  let mut xs     = alloc.list.new<i32>(&mut sram, 16)?
  alloc.list.push(&mut xs, 42)?
  alloc.list.push(&mut xs,  7)?
  return alloc.list.len(&xs) as i32
}
```

The same `alloc.list.new`, `push`, `len` work here as in the arena
case. No fork of stdlib for embedded — that is the whole point of the
parameterized design. An embedded build can omit libc linkage entirely
and still use `alloc.*` containers, provided it doesn't bring in
`mem.arena` (which calls `malloc`).

### `std.io.read_file<A>` — type-level allocator parameter

```
// std/io.qs
module std.io {
  import mem.{Allocator, AllocError}
  import core.result.{Result, Ok, Err}
  import core.str.String

  // No module-level `wirable A` — different callers want different
  // allocators per-call (read_file into a scratch arena, log into a
  // long-lived gpa, etc.). Type-level parameter instead.

  enum IoError { NotFound, PermissionDenied, ReadFailed, AllocFailed }

  fn read_file<A: Allocator>(path: *u8, alloc: *A) -> Result<String, IoError> {
    let fd = open(path, 0)
    if fd < 0 { return Err(IoError::NotFound) }
    defer close(fd)

    let size = lseek(fd, 0, 2)
    lseek(fd, 0, 0)

    let buf = match alloc.alloc(size) {
      Ok(p)  => p,
      Err(_) => return Err(IoError::AllocFailed),
    }
    let n_read = read(fd, buf, size)
    if n_read != size { return Err(IoError::ReadFailed) }
    Ok(String { ptr: buf, len: size })
  }
}
```

The choice between wirable and type-level here: `std.*` modules use
type-level `<A>` because different syscalls in the same program want
different backing allocators. Binding once-per-import would be too
coarse. Library author judgement; both forms are first-class.

## Tradeoffs

- **Code size.** Realistic ballpark: 50-200 fresh struct instantiations
  in a non-trivial program. Manageable but real.
- **Inference at use sites.** A library function over `List<T, _>`
  propagates `_` through its callers until something binds it. Zig and
  Rust both hit this; the cost is mostly in error messages and
  inference complexity, not in expressivity.
- **Migration churn.** Every container author wires their own wirable
  through inner imports. Verbose but explicit.

## Not yet built

- **Per-symbol import.** `import List from alloc.list` shape. Lands
  after whole-module import + wiring works.
- **Implicit transitive parameter propagation.** Considered; rejected
  for "explicit > implicit." Library authors re-declare and forward.
- **Default allocator at program root.** Considered; rejected — wiring
  at import is the resolution seam, no global default.
- **Comptime evaluation.** Out of current scope.
- **Claim-erased generics** (a `T` represented as `*u8` plus a claim
  `is_a<T>`). Research direction; out of current scope.
- **`From`-trait error coercion** for nested `?` across error types.
- **`defer` / `Drop` trait.** `quod.with_arena` is the only scope-exit
  mechanism today and is hardcoded to one allocator. Generalizing
  requires either a `Drop` trait or `defer` syntax. Without it, error
  paths in fallible functions need explicit cleanup at every `return`.
- **`catch { }` block** as sugar for
  `match { Ok(v) => v, Err(_) => ... }`. Pure ergonomics.

## Open design questions

- **Self type in trait methods.** Today's impls re-declare the receiver
  type explicitly (`fn alloc(self: *Arena, ...)`). `SelfType` inside
  trait method declarations is rewritten to `for_type` at impl
  construction. Considered: making `Self` a magic type in user code.
- **`&mut a` vs raw pointer receiver.** Today every reference is a raw
  `*T`; there is no `&` / `&mut` mutability tracking. Surface sugar with
  mutability tracking is a separate multi-month feature.
- **UFCS vs methods for trait dispatch.** `arena.alloc(n)` reads better
  than `Allocator::alloc(&arena, n)` but requires method resolution
  that respects the trait bound. Today's quod is UFCS. The mechanic
  isn't hard; just commit one way.
- **Const generics** (`FixedBuffer<const N: usize>`). Probably defer;
  runtime-cap is fine and matches Zig's model.
- **Re-wiring across imports.** The implementation correctly handles
  `import alloc.json wire A=Arena` propagating into `alloc.json`'s
  embedded `import alloc.list wire A=A`, but the resolution algorithm
  is worth re-sketching when adding new wiring features.
