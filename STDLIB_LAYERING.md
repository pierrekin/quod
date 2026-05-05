# quod — stdlib namespace layering

How the stdlib is organized into layers. Four namespaces (`core`, `mem`,
`alloc`, `std`) decided by three orthogonal questions about what a
module needs at runtime. Read `LANGUAGE.md` for the user-facing tier
description; this document is the placement rule for new modules.

## The four namespaces

**`core.*`** — pure compute. No allocator, no OS, no syscalls. Pure
types, pure functions, and pure trait declarations *unrelated to
memory*. Usable in a no-runtime program.

**`mem.*`** — memory providers. Hosts the `Allocator` trait and every
concrete provider (`Arena`, `Gpa`, `Page`, `FixedBuffer`). Open set:
third-party and application code can add new providers without touching
the stdlib.

**`alloc.*`** — consumers that need an allocator. Every public type is
abstract over `A: mem.Allocator` (`List<T, A>`, `Map<K, V, A>`,
`Builder<A>`, `JsonParser<A>`, …). Useless without an `A` wired in, but
does not statically depend on any concrete provider.

**`std.*`** — needs an OS. File and process I/O, time, env, args. Most
`std.*` modules also take an `A` — the OS-dependency is the gating
axis, allocator-dependency is incidental.

## Litmus tests

Three orthogonal questions decide the layer:

| question | yes → | no → |
|---|---|---|
| Does this need an OS? | `std.*` | next |
| Does this need an allocator? | `alloc.*` | next |
| Is it a memory provider? | `mem.*` | `core.*` |

**`core` litmus**: *can I write a meaningful program using only this
module and zero `Allocator` instances?* If yes → `core`. `Option<T>`,
`Result<T, E>`, `String` (slice), `bytes.eq`, `cmp.Ord` all pass.
`List<T>` fails — you cannot construct one without an `A`, even an
abstract one.

**`mem` litmus**: *does this module declare or implement the
`Allocator` trait?* The trait itself, every concrete allocator, and
combinators like `mem.tracking.Tracking<Inner>` (an instrumented
wrapper) live here.

**`alloc` litmus**: *does the module have a `mem.Allocator`-shaped
parameter on its public types?* If yes → `alloc`. The `A` parameter is
the namespace marker — encoded once in the type signature, again in the
directory.

## Why the trait lives in `mem`, not `core`

A `core`-only program by definition has no allocator. It never needs
to *reference* the `Allocator` trait. Pulling the trait down to `core`
adds dependency surface for no gain. Colocating with the providers it
constrains is cleaner: `mem.Allocator`, `mem.arena.Arena`, `mem.gpa.Gpa`
all sit together.

Other trait declarations (`core.cmp.Ord`, `core.iter.Iter`,
`core.hash.Hash`) stay in `core` because they are not about memory and
are usable without one.

## Dependency graph

```
              std    (OS + (usually) allocator)
              ↓
              alloc  (consumers, abstract over A)
              ↓
              core   (pure compute)
              ↑
              mem    (providers — peer of alloc, not below it)
```

Edges:

- `mem` depends on `core` (for byte ops, slice types).
- `alloc` depends on `core` and on `mem.Allocator` (the trait only).
- `std` depends on `core` and on `mem.Allocator` (the trait only); may
  also use `alloc.*` consumers internally.
- **`alloc` does not statically depend on any concrete provider in
  `mem`.** This is the key property: libraries stay pure, applications
  are the only place that imports both `alloc` and a concrete provider.

Apps are the only place where everything resolves. They import
`alloc.list`, import `mem.arena`, and wire `A=mem.arena.Arena` at the
import site (per `MODULE_SYSTEM.md`'s `wire` syntax).

## Providers are an open set

`mem.*` is not a closed list of stdlib-blessed allocators. Anything
that implements `mem.Allocator` is a valid provider. Concretely:

```
struct SramAllocator { buf: *u8, cap: i64, cursor: i64 }

impl mem.Allocator for SramAllocator {
  fn alloc(self: *Self, n: i64) -> Result<*u8, AllocError> { ... }
  fn free (self: *Self, p: *u8)                            { ... }
}
```

An embedded user with 4 KB of SRAM can use the *same* `alloc.list.List`,
`alloc.str.Builder`, `alloc.json.Parser` we ship — they bring their own
`SramAllocator`, wire it at import, and the monomorphizer produces
SRAM-flavored bodies of those types. No fork of the stdlib, no special
"embedded" namespace.

The stdlib commits to shipping a small set of common providers:

- `mem.arena.Arena` — bump allocator, free-on-drop, multi-chunk.
- `mem.fixed.FixedBuffer` — bump allocator over a borrowed slice;
  never grows past the slice. Validates the embedded contract.
- `mem.gpa.Gpa` — general-purpose allocator (size classes + free
  list). Future.
- `mem.page.Page` — page-aligned mmap allocator. Future.

## Fallible allocation → `Result` on `push`

A consequence of provider-openness: a fixed-buffer allocator runs out
of space; an arena's `mmap` may fail. So `mem.Allocator::alloc` returns
`Result<*u8, AllocError>`, and every consumer operation that allocates
threads that result through:

```
fn push(self: *List<T, A>, x: T) -> Result<(), AllocError>
fn push_byte(self: *Builder<A>, b: u8) -> Result<(), AllocError>
```

This is the right outcome: no consumer assumes the allocator can't
fail. The trait forces fallibility through the API.

## Migration status

The v1 stdlib (allocator-as-value, `core.*` and `alloc.*` only) has
been migrated to this layering. State:

| target                                            | status |
|---------------------------------------------------|--------|
| `core.bytes`                                      | ✅ shipped |
| `core.str.String` (slice)                         | ✅ shipped |
| `core.option.Option<T>`                           | ✅ shipped |
| `core.result.Result<T, E>`                        | ✅ shipped |
| `mem.Allocator` (trait)                           | ✅ shipped |
| `mem.arena.Arena`                                 | ✅ shipped |
| `mem.fixed.FixedBuffer`                           | ✅ shipped |
| `alloc.list.List<T, A>`                           | ✅ shipped |
| `alloc.str.Builder<A>`                            | ✅ shipped |
| `alloc.str.to_cstr_in<A>`                         | ✅ shipped |
| `alloc.json.Parser<A>`                            | ✅ shipped |
| `alloc.json.write.write_value<A>`                 | ✅ shipped |
| `alloc.map.Map<K, V, A>`                          | ⏳ not yet |
| `std.io.read_file<A>`                             | ⏳ not yet (still `read_file_to_arena`) |
| `mem.gpa.Gpa`                                     | ⏳ future |
| `mem.page.Page`                                   | ⏳ future |

## What this does not decide

- **Wire syntax** — see `MODULE_SYSTEM.md` (`import X wire P=Y`,
  module-level vs type-level params, `as`-aliasing for re-wiring).
- **Generics machinery** — see `MODULE_SYSTEM.md`. Monomorphization
  pass already shipped.
- **Erasure seam** — `dyn Alloc`-style boundaries for cases where you
  genuinely need allocator-agnostic functions. Sketched in
  `MODULE_SYSTEM.md`; concrete shape TBD.

## Why four namespaces is still "three layers"

The user-visible mental model stays three:

1. *Does my program need memory?* → reach for `alloc`.
2. *Does my program need an OS?* → reach for `std`.
3. *Otherwise?* → `core`.

`mem` is sideways from `alloc`, not above or below. You don't reach for
`mem` to express *behavior* — you reach for it to wire a concrete
provider into something from `alloc` or `std`. Library code never
imports a concrete `mem.*` module; only application code does.
