# JSON parser — v1 case study

Seven snapshots, one per phase, walking from project skeleton to a full
recursive-descent JSON parser. Every phase is its own buildable
program: `quod -p json_v1_phaseN show`, `quod -p json_v1_phaseN run`.

The case study is the asset. v1 is intentionally awkward — fat-struct
`JsonValue` (one struct, ten fields, six unused per variant) and an
out-param + bool-return calling convention everywhere — so each phase
adds friction we can later use as the requirements doc for sum types.

| phase | what it adds | what it teaches |
| --- | --- | --- |
| 0 | project skeleton | how a quod program starts: empty `main`, no functions, no constants |
| 1 | byte primitives (`peek_byte`, `at_end`, `consume_byte`, `expect_byte`) | reading bytes through an i8\* parser handle; cursor advance via load+store |
| 2 | keyword parsers (`parse_null`, `parse_true`, `parse_false`) + whitespace | the `if !ok return false` pyramid is born; arena-allocated parsers and value buffers |
| 3 | `parse_number` (signed i64) | first real digit-loop in quod-script; type-suffix integer literals (`1i8` etc.) earn their keep |
| 4 | `parse_string` with `\"\\\\\\/\\n\\t\\r\\b\\f` escapes | first variable-sized arena allocation; worst-case sized output buffer |
| 5 | `parse_value` dispatcher | first-byte routing across the five scalars; whitespace skip in front |
| 6 | `parse_array` | first recursion through `parse_value`; pointer-arithmetic indexing across a stride that should be `sizeof(JsonValue)` but is hard-coded to 128 |
| 7 | `parse_object` | parallel arrays for keys/values; `string : value` with whitespace tolerance |

## Friction log

Held intentionally — these are the design inputs for v2 (sum types):

- **Fat union.** `JsonValue` carries `tag, b, n, str_ptr, str_len, arr_ptr, arr_count, obj_keys, obj_values, obj_count` even when the actual variant uses 1–3 fields. Each `parse_*` writes only its own slots. Wasted memory, wasted code, no exhaustiveness check on `tag`.
- **Out-param + bool-return.** Every parser is `fn parse_X(p, out) -> i1`. Failure returns 0 with `parser.had_error` set; success leaves the result at `out`. Stack pyramid: `let ok = parse_X(...); if (ok == 0) { return 0 }`. Six lines per call site.
- **No `sizeof`.** Element strides are hardcoded to 128 (the alloc size we use in `main`). If `JsonValue` ever changes, every array/object index breaks silently.
- **No early-return short-circuit.** `quod.try` was sketched in `.scratch/thoughts.md` but deferred — it would compile to the v1 calling convention and bake it deeper rather than pave over it.
- **Field-set type narrowing.** quod-script doesn't know struct field types, so `parser.had_error = 1` defaults to i64 and fails to lower. You must write `1i8` explicitly. A pain in v1 main, much smaller after sum types remove ad-hoc tag bytes.

## Reading the commits

Each phase is its own commit; the message narrates what was added and
what the friction was. `git log --oneline -- examples/json/` shows the
through-line.
