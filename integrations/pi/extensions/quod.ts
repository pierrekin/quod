// Pi extension exposing the quod CLI as a set of tools.
//
// Drop into a project's .pi/extensions/, or load ad-hoc with:
//   pi -e .pi/extensions/quod.ts
//
// Every tool shells out to the `quod` binary on $PATH. Tools that touch a
// project accept `cwd` (the directory containing quod.toml) and `program`
// (to pick a [[program]] in a workspace). Inspection tools also accept
// `program_file` for `quod -f PATH ...` — point at any program.json
// directly, bypassing quod.toml (handy for stdlib modules).
//
// For deeper context: tool descriptions are short by design. The agent
// should call the relevant `*_ls` / `quod_schema` / read GUIDE.md /
// LANGUAGE.md / DEVELOPING.md / `quod <verb> --help` as needed.

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { StringEnum } from "@mariozechner/pi-ai";
import { execFile, type ChildProcess } from "node:child_process";
import { Type } from "typebox";

interface RunOpts {
  cwd?: string;
  stdin?: string;
  signal?: AbortSignal;
}

function runQuod(args: string[], opts: RunOpts = {}): Promise<string> {
  return new Promise((resolve, reject) => {
    const child: ChildProcess = execFile(
      "quod",
      args,
      {
        cwd: opts.cwd,
        signal: opts.signal,
        maxBuffer: 16 * 1024 * 1024,
      },
      (err, stdout, stderr) => {
        if (err) {
          const msg = (stderr || "").trim() || err.message;
          reject(new Error(`quod ${args.join(" ")} failed: ${msg}`));
          return;
        }
        const tail = stderr ? `\n[stderr]\n${stderr}` : "";
        resolve((stdout || "") + tail);
      },
    );
    if (opts.stdin !== undefined && child.stdin) {
      child.stdin.end(opts.stdin);
    }
  });
}

function text(s: string) {
  return { content: [{ type: "text" as const, text: s }] };
}

// Selector fields threaded through every tool that touches a project.
const selectorFields = {
  cwd: Type.Optional(
    Type.String({
      description:
        "Project directory (containing quod.toml). Defaults to pi's working directory.",
    }),
  ),
  program: Type.Optional(
    Type.String({
      description:
        "Select a [[program]] in a workspace quod.toml. Omit when only one program is configured.",
    }),
  ),
};

// Inspection tools additionally accept program_file (-f), which bypasses
// quod.toml entirely — useful for inspecting a standalone program.json
// (e.g. files under src/quod/stdlib/). Mutually exclusive with cwd/program.
const inspectionFields = {
  ...selectorFields,
  program_file: Type.Optional(
    Type.String({
      description:
        "Path to a standalone program.json. Bypasses quod.toml. Useful for inspecting stdlib modules in src/quod/stdlib/.",
    }),
  ),
};

// Prepend the global selector flags before the subcommand verb.
function withSelectors(p: { program?: string; program_file?: string }, sub: string[]): string[] {
  const head: string[] = [];
  if (p.program_file) head.push("-f", p.program_file);
  if (p.program) head.push("-p", p.program);
  return [...head, ...sub];
}

const fnRefField = {
  function: Type.String({
    description: "Function name or content-hash prefix.",
  }),
};

const claimKind = StringEnum(["non_negative", "int_range", "return_in_range"] as const);
const regime = StringEnum(["axiom", "witness"] as const);
const enforcement = StringEnum(["trust", "verify"] as const);
const scalarType = StringEnum(["i1", "i8", "i16", "i32", "i64", "i8_ptr"] as const);

export default function (pi: ExtensionAPI): void {
  // -------------------- lifecycle --------------------

  pi.registerTool({
    name: "quod_init",
    label: "Init quod project",
    description:
      "Initialize a new quod project: writes quod.toml and program.json. Templates: hello (runnable hello-world), guarded (claim/proof playground with an unproven function f), empty.",
    promptSnippet: "Initialize a new quod project (writes quod.toml + program.json).",
    promptGuidelines: [
      "When starting a new quod project, call quod_init. Pick template=hello for a runnable starter, guarded for a claim/proof playground, empty for a blank slate.",
      "After init, typical next step depends on the goal: quod_show to inspect, quod_run to compile-and-execute (hello), or quod_fn_unconstrained → quod_claim_suggest to start the optimization workflow (guarded).",
    ],
    parameters: Type.Object({
      cwd: selectorFields.cwd,
      template: StringEnum(["hello", "guarded", "empty"] as const),
      force: Type.Optional(
        Type.Boolean({ description: "Overwrite existing files." }),
      ),
    }),
    async execute(_id, p, signal) {
      const args = ["init", "-t", p.template];
      if (p.force) args.push("--force");
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_ingest",
    label: "Ingest C source",
    description:
      "Bootstrap a new quod project from a C source file. Writes quod.toml + program.json in cwd; refuses if either already exists. The supported C subset is intentionally narrow (int-only, no structs/floats/for/switch); refusals raise IngestError with a source location.",
    parameters: Type.Object({
      cwd: selectorFields.cwd,
      source: Type.String({ description: "Path to a .c source file (relative to cwd or absolute)." }),
      name: Type.Optional(
        Type.String({ description: "Program name in quod.toml. Defaults to the source file's stem." }),
      ),
      imports: Type.Optional(
        Type.Array(Type.String(), {
          description: "Stdlib modules to add to the program's imports list, e.g. ['core.str', 'std.io'].",
        }),
      ),
    }),
    async execute(_id, p, signal) {
      const args = ["ingest", p.source];
      if (p.name) args.push("-n", p.name);
      if (p.imports) for (const m of p.imports) args.push("--import", m);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_check",
    label: "Check program",
    description:
      "Parse, lower, and LLVM-verify the program. No artifacts emitted. Use this as a fast sanity check after edits.",
    parameters: Type.Object({ ...selectorFields }),
    async execute(_id, p, signal) {
      return text(await runQuod(withSelectors(p, ["check"]), { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_build",
    label: "Build binaries",
    description:
      "Lower, optimize, emit objects, and link a binary for every [[program.bin]] in quod.toml. Tier flags: no_std refuses imports from std.* (OS-dependent); no_alloc refuses alloc.* and std.* (and refuses with_arena) — bare-metal mode. Enforcement overrides change how claims of each regime lower: 'trust' = llvm.assume (UB if false), 'verify' = runtime branch + abort.",
    parameters: Type.Object({
      ...selectorFields,
      profile: Type.Optional(
        Type.Number({ description: "LLVM optimization level 0..3. 0 skips the optimize pass." }),
      ),
      target: Type.Optional(Type.String({ description: "LLVM target triple. Defaults to host." })),
      link: Type.Optional(
        Type.Boolean({ description: "Link object files into a binary. Defaults to quod.toml [build].link." }),
      ),
      show_ir: Type.Optional(
        Type.Boolean({ description: "Print optimized IR to stdout." }),
      ),
      no_std: Type.Optional(Type.Boolean({ description: "Refuse to resolve imports from std.*." })),
      no_alloc: Type.Optional(
        Type.Boolean({ description: "Refuse alloc.* and std.* imports; refuse with_arena." }),
      ),
      enforce_axiom: Type.Optional(enforcement),
      enforce_witness: Type.Optional(enforcement),
      enforce_lattice: Type.Optional(enforcement),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["build"]);
      if (p.profile !== undefined) args.push("--profile", String(p.profile));
      if (p.target) args.push("--target", p.target);
      if (p.link === true) args.push("--link");
      else if (p.link === false) args.push("--no-link");
      if (p.show_ir) args.push("--show-ir");
      if (p.no_std) args.push("--no-std");
      if (p.no_alloc) args.push("--no-alloc");
      if (p.enforce_axiom) args.push("--enforce-axiom", p.enforce_axiom);
      if (p.enforce_witness) args.push("--enforce-witness", p.enforce_witness);
      if (p.enforce_lattice) args.push("--enforce-lattice", p.enforce_lattice);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_run",
    label: "Build and run",
    description:
      "Build the project and execute one of its [[program.bin]] entries. Captures stdout, stderr, and exit code. If the entry function declares int params, pass them via `program_args`; the synthesized main wrapper parses each via atoll then trunc/sext's to the param's width.",
    parameters: Type.Object({
      ...selectorFields,
      bin: Type.Optional(
        Type.String({
          description:
            "Which [[program.bin]] to run. Required if multiple bins are configured.",
        }),
      ),
      program_args: Type.Optional(
        Type.Array(Type.String(), {
          description:
            "Args forwarded to the spawned binary as argv. For an entry with N int params, pass N integer-shaped strings (atoll-parsed at the wrapper, then trunc/sext'd to each param's width).",
        }),
      ),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["run"]);
      if (p.bin) args.push("--bin", p.bin);
      if (p.program_args && p.program_args.length > 0) {
        args.push("--", ...p.program_args);
      }
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  // -------------------- schema introspection --------------------

  pi.registerTool({
    name: "quod_schema",
    label: "Show node schema",
    description:
      "Discover the JSON shape of any node. With no args, lists categories (statement, expression, type, claim, justification, program). With `category`, lists kinds in that category as one-liners. With `kind`, returns the full schema: required/optional fields with types, plus a minimal example. ALWAYS call this before constructing JSON program nodes — saves round-trips that otherwise fail validation. Read-only; needs no project.",
    parameters: Type.Object({
      kind: Type.Optional(
        Type.String({
          description:
            "Node kind, e.g. 'quod.let', 'quod.match', 'quod.try', 'llvm.binop', 'int_range', 'Function', 'EnumDef', 'StructDef'.",
        }),
      ),
      category: Type.Optional(
        StringEnum([
          "expression",
          "statement",
          "type",
          "claim",
          "justification",
          "program",
        ] as const),
      ),
    }),
    async execute(_id, p, signal) {
      const args = ["schema"];
      if (p.kind) args.push(p.kind);
      else if (p.category) args.push("--category", p.category);
      return text(await runQuod(args, { signal }));
    },
  });

  // -------------------- whole-program inspection --------------------

  pi.registerTool({
    name: "quod_show",
    label: "Show program",
    description:
      "Print the program in canonical form with content-hash prefixes. Pass hashes=true to dump every node and its short hash instead. Pass program_file to inspect a standalone program.json without a quod.toml — handy for the stdlib modules under src/quod/stdlib/.",
    parameters: Type.Object({
      ...inspectionFields,
      hashes: Type.Optional(Type.Boolean()),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["show"]);
      if (p.hashes) args.push("--hashes");
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_find",
    label: "Find by hash prefix",
    description:
      "Resolve a content-hash prefix to a node and print its hash, type, and JSON.",
    parameters: Type.Object({
      ...inspectionFields,
      prefix: Type.String({ description: "A unique content-hash prefix." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["find", p.prefix]), { cwd: p.cwd, signal }),
      );
    },
  });

  // -------------------- fn --------------------

  pi.registerTool({
    name: "quod_fn_ls",
    label: "List functions",
    description: "List all functions with their signatures and content hashes.",
    parameters: Type.Object({ ...inspectionFields }),
    async execute(_id, p, signal) {
      return text(await runQuod(withSelectors(p, ["fn", "ls"]), { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_fn_show",
    label: "Show one function",
    description:
      "Print a single function (body, claims, notes). Accepts a name or hash prefix.",
    parameters: Type.Object({ ...inspectionFields, ref: Type.String() }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["fn", "show", p.ref]), { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_fn_add_script",
    label: "Add function (script)",
    description:
      "Append a new function written in quod-script — a compact textual surface that emits the same JSON nodes as quod_fn_add. PREFERRED for non-trivial bodies. Grammar covers let/if/while/for/return/store/match/with_arena statements, ?-propagation, sizeof[T], widen/uwiden/load/ptr_offset, struct and enum literals (Type { f: e } / Enum::Variant { f: e }), calls (dotted names supported, e.g. core.str.eq), all binops, &&/||. Out of scope: claims and any top-level declarations (struct/enum/extern/const/import) — those are separate verbs.",
    parameters: Type.Object({
      ...selectorFields,
      script: Type.String({
        description:
          "Inline quod-script source, e.g. 'fn clamp(x: i64, lo: i64, hi: i64) -> i64 { if (x < lo) { return lo } if (x > hi) { return hi } return x }'.",
      }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["fn", "add", "--script", p.script]), {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });

  pi.registerTool({
    name: "quod_fn_add",
    label: "Add function (JSON)",
    description:
      "Append a new function from a JSON Function spec. Use this when you need to attach claims/notes inline, or anything outside the script grammar; otherwise prefer quod_fn_add_script. Call quod_schema(kind='Function') for the canonical shape, or quod_schema(category='statement') / (category='expression') for the body's building blocks.",
    parameters: Type.Object({
      ...selectorFields,
      spec_json: Type.String({
        description: "JSON Function object. See quod_schema(kind='Function').",
      }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["fn", "add", "-"]), {
          cwd: p.cwd,
          stdin: p.spec_json,
          signal,
        }),
      );
    },
  });

  pi.registerTool({
    name: "quod_fn_rename",
    label: "Rename function",
    description:
      "Rename a function and update every call site that names it. Refuses if the new name already exists.",
    parameters: Type.Object({
      ...selectorFields,
      old_name: Type.String(),
      new_name: Type.String(),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["fn", "rename", p.old_name, p.new_name]), {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });

  pi.registerTool({
    name: "quod_fn_rm",
    label: "Remove function",
    description:
      "Remove a function from the program. Permissive: doesn't refuse if other functions still call this one — the dangling call surfaces at build time. Use quod_fn_callers first to see who'd be affected.",
    parameters: Type.Object({ ...selectorFields, ...fnRefField }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["fn", "rm", p.function]), { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_fn_callers",
    label: "Find callers",
    description: "List every call site to a function across the program.",
    parameters: Type.Object({
      ...inspectionFields,
      target: Type.String({ description: "Function name to find callers of." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["fn", "callers", p.target]), { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_fn_data_flow",
    label: "Param data flow",
    description:
      "Show every statement in `function` that reads `param`. Useful for understanding how a parameter is used.",
    parameters: Type.Object({
      ...inspectionFields,
      ...fnRefField,
      param: Type.String({ description: "Parameter name." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["fn", "data-flow", p.function, p.param]), {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });

  pi.registerTool({
    name: "quod_fn_call_graph",
    label: "Call graph",
    description:
      "Print the static call graph: caller → callees, plus roots and leaves. Externs are tagged @extern; dangling callees with !.",
    parameters: Type.Object({ ...inspectionFields }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["fn", "call-graph"]), { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_fn_unconstrained",
    label: "Unconstrained params",
    description:
      "List parameters that have no claim attached. A scout for where claims could be added.",
    parameters: Type.Object({ ...inspectionFields }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["fn", "unconstrained"]), { cwd: p.cwd, signal }),
      );
    },
  });

  // -------------------- claim --------------------

  pi.registerTool({
    name: "quod_claim_ls",
    label: "List claims",
    description:
      "List stored claims (axiom + witness regimes). Pass `function` to restrict to one function.",
    parameters: Type.Object({
      ...inspectionFields,
      function: Type.Optional(Type.String()),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["claim", "ls"]);
      if (p.function) args.push(p.function);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_add",
    label: "Add claim",
    description:
      "Attach a claim to a function. The optimizer will trust this assertion. Use regime=axiom (you assert) or witness (proven). non_negative and int_range need `target`; return_in_range must omit it.",
    promptSnippet: "Attach a claim — axiom (you assert) or witness (proven).",
    promptGuidelines: [
      "Use quod_claim_add for facts you can assert without proof, e.g. when the user has told you a parameter is non-negative. The optimizer trusts axiom claims; behavior is undefined at runtime if they're violated.",
      "Prefer quod_claim_prove over quod_claim_add when the claim should be derivable from the function's body — Z3 will check it and attach a hash-pinned witness, which is safer than asserting blindly.",
    ],
    parameters: Type.Object({
      ...selectorFields,
      ...fnRefField,
      kind: claimKind,
      target: Type.Optional(
        Type.String({
          description:
            "Parameter name. Required for non_negative and int_range; must be omitted for return_in_range.",
        }),
      ),
      min: Type.Optional(Type.Number()),
      max: Type.Optional(Type.Number()),
      regime: Type.Optional(regime),
      enforcement: Type.Optional(enforcement),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["claim", "add", p.function, p.kind]);
      if (p.target) args.push(p.target);
      if (p.min !== undefined) args.push("--min", String(p.min));
      if (p.max !== undefined) args.push("--max", String(p.max));
      if (p.regime) args.push("--regime", p.regime);
      if (p.enforcement) args.push("--enforcement", p.enforcement);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_relax",
    label: "Remove claim",
    description: "Remove a claim (always safe — drops an assertion).",
    parameters: Type.Object({
      ...selectorFields,
      ...fnRefField,
      kind: claimKind,
      target: Type.Optional(Type.String()),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["claim", "relax", p.function, p.kind]);
      if (p.target) args.push(p.target);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_prove",
    label: "Prove claim with Z3",
    description:
      "Synthesize an SMT-LIB encoding of the claim, run a witness provider, and on success attach the result as a witness claim with a hash-pinned .smt2 artifact. Default provider is the first registered witness/prove provider (z3.qf_lia). Pass `provider` to pick a specific one — see quod_provider_ls.",
    promptSnippet: "Discharge a claim via Z3; attach as witness on success.",
    promptGuidelines: [
      "Use quod_claim_prove to formally verify a claim. On success the proof is stored as a .smt2 artifact and the claim is attached with regime=witness.",
      "If proof returns 'sat', Z3 found a counterexample — the claim is false. Do NOT fall back to quod_claim_add as axiom; revisit the claim or the function.",
      "If proof returns 'unknown' or NotImplementedError, the claim is beyond the SMT lowering (mutable locals, srem, unsigned cmps). Either refactor the function into a pure-expression form or skip proving that particular claim.",
    ],
    parameters: Type.Object({
      ...selectorFields,
      ...fnRefField,
      kind: claimKind,
      target: Type.Optional(Type.String()),
      min: Type.Optional(Type.Number()),
      max: Type.Optional(Type.Number()),
      enforcement: Type.Optional(enforcement),
      provider: Type.Optional(
        Type.String({
          description: "Provider name (e.g. 'z3.qf_lia'). Defaults to the first witness/prove provider.",
        }),
      ),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["claim", "prove", p.function, p.kind]);
      if (p.target) args.push(p.target);
      if (p.min !== undefined) args.push("--min", String(p.min));
      if (p.max !== undefined) args.push("--max", String(p.max));
      if (p.enforcement) args.push("--enforcement", p.enforcement);
      if (p.provider) args.push("--provider", p.provider);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_verify",
    label: "Verify claim evidence",
    description:
      "Re-check evidence attached to stored claims: re-hashes z3 artifacts and re-runs Z3 to confirm unsat.",
    promptSnippet: "Re-validate stored proofs after edits.",
    promptGuidelines: [
      "Run quod_claim_verify after editing a function that has witness claims. The .smt2 artifact's sha256 is checked, and Z3 re-runs to confirm unsat. If a proof breaks, you'll need to re-prove with quod_claim_prove or relax the claim.",
    ],
    parameters: Type.Object({ ...selectorFields }),
    async execute(_id, p, signal) {
      return text(await runQuod(withSelectors(p, ["claim", "verify"]), { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_suggest",
    label: "Suggest claims",
    description:
      "Speculatively compile candidate claims and surface those that would shrink optimized IR if proven. Read-only — does not mutate the program.",
    promptSnippet:
      "Find claims worth proving — entry point of the optimization workflow.",
    promptGuidelines: [
      "When the user asks to optimize a quod program (or 'make it faster', 'reduce IR size'), run quod_claim_suggest first. It speculatively compiles candidate claims and reports which ones would shrink optimized IR if proven.",
      "After quod_claim_suggest, run quod_claim_prove on the suggestions that should genuinely hold. Don't fall back to quod_claim_add as axiom unless you've verified the claim by other means — axioms are trusted unconditionally and behavior is undefined if violated.",
      "If the suggester reports nothing, the codegen is already tight or the candidate set is exhausted; not every program benefits from claim-driven optimization.",
    ],
    parameters: Type.Object({
      ...selectorFields,
      top_n: Type.Optional(Type.Number()),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["claim", "suggest"]);
      if (p.top_n !== undefined) args.push("--top-n", String(p.top_n));
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_derive",
    label: "Derive lattice claims",
    description:
      "Run the lattice analysis and print derived (regime=lattice) claims. These are re-derived every compile and never stored.",
    parameters: Type.Object({ ...inspectionFields }),
    async execute(_id, p, signal) {
      return text(await runQuod(withSelectors(p, ["claim", "derive"]), { cwd: p.cwd, signal }));
    },
  });

  // -------------------- stmt --------------------

  pi.registerTool({
    name: "quod_stmt_add",
    label: "Add statement",
    description:
      "Insert a statement into a function. Provide the statement as a JSON Statement object in `spec_json`. Anchor selects where to insert: at_end, at_start, before <hash>, after <hash>. Call quod_schema(category='statement') for valid statement kinds and their fields.",
    parameters: Type.Object({
      ...selectorFields,
      ...fnRefField,
      spec_json: Type.String({
        description: "JSON Statement object. See quod_schema(category='statement').",
      }),
      anchor: StringEnum(["at_end", "at_start", "before", "after"] as const),
      anchor_ref: Type.Optional(
        Type.String({
          description:
            "Hash prefix of an existing statement. Required when anchor is `before` or `after`.",
        }),
      ),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["stmt", "add", p.function, "-"]);
      if (p.anchor === "at_end") args.push("--at-end");
      else if (p.anchor === "at_start") args.push("--at-start");
      else if (p.anchor === "before") {
        if (!p.anchor_ref) throw new Error("anchor=before requires anchor_ref");
        args.push("--before", p.anchor_ref);
      } else if (p.anchor === "after") {
        if (!p.anchor_ref) throw new Error("anchor=after requires anchor_ref");
        args.push("--after", p.anchor_ref);
      }
      return text(
        await runQuod(args, { cwd: p.cwd, stdin: p.spec_json, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_stmt_rm",
    label: "Remove statement",
    description:
      "Remove a statement from a function by content-hash prefix. Find the hash via quod_fn_show (each statement is shown with its short hash) or quod_show with hashes=true.",
    parameters: Type.Object({
      ...selectorFields,
      ...fnRefField,
      hash_prefix: Type.String({
        description: "Content-hash prefix of the statement to remove.",
      }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["stmt", "rm", p.function, p.hash_prefix]), {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });

  // -------------------- const --------------------

  pi.registerTool({
    name: "quod_const_ls",
    label: "List constants",
    description: "List declared string constants.",
    parameters: Type.Object({ ...inspectionFields }),
    async execute(_id, p, signal) {
      return text(await runQuod(withSelectors(p, ["const", "ls"]), { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_const_add",
    label: "Add string constant",
    description:
      "Declare a string constant. Reference it from code with quod.string_ref. Pass the value as a raw string (with literal newlines etc.); quod adds a trailing NUL when lowering.",
    parameters: Type.Object({
      ...selectorFields,
      name: Type.String({ description: "Constant name, e.g. '.str.fmt'." }),
      value: Type.String({ description: "Raw string value (not C-escaped)." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["const", "add", p.name, p.value]), {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });

  pi.registerTool({
    name: "quod_const_rm",
    label: "Remove string constant",
    description:
      "Remove a string constant. Permissive: doesn't refuse if a quod.string_ref still points at it — the dangling reference surfaces at build time.",
    parameters: Type.Object({
      ...selectorFields,
      name: Type.String({ description: "Constant name to remove." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["const", "rm", p.name]), { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_const_rename",
    label: "Rename string constant",
    description: "Rename a string constant and update every quod.string_ref to it.",
    parameters: Type.Object({
      ...selectorFields,
      old_name: Type.String(),
      new_name: Type.String(),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["const", "rename", p.old_name, p.new_name]), {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });

  // -------------------- struct --------------------

  pi.registerTool({
    name: "quod_struct_ls",
    label: "List structs",
    description: "List declared structs with their field signatures.",
    parameters: Type.Object({ ...inspectionFields }),
    async execute(_id, p, signal) {
      return text(await runQuod(withSelectors(p, ["struct", "ls"]), { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_struct_show",
    label: "Show one struct",
    description: "Print one struct definition with field names and types.",
    parameters: Type.Object({
      ...inspectionFields,
      name: Type.String({ description: "Struct name." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["struct", "show", p.name]), { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_struct_add",
    label: "Add struct",
    description:
      "Define a new struct. Field types are int widths (i1/i8/i16/i32/i64), i8_ptr, or any struct already defined in the program. The model validator catches dangling refs and cycles before the file is written.",
    parameters: Type.Object({
      ...selectorFields,
      name: Type.String({ description: "Struct name (e.g. 'Point', 'Parser')." }),
      fields: Type.Array(
        Type.String({
          description: "Field as 'name:type', e.g. 'x:i32', 'cur:i8_ptr', 'inner:Point'.",
        }),
        { description: "One or more 'name:type' tokens." },
      ),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["struct", "add", p.name, ...p.fields]);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_struct_rm",
    label: "Remove struct",
    description:
      "Remove a struct definition. Strict: refuses if anything in the program (other structs, function params/locals, struct_init, field reads) references it. Use quod_show to find references first.",
    parameters: Type.Object({
      ...selectorFields,
      name: Type.String({ description: "Struct name to remove." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["struct", "rm", p.name]), { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_struct_rename",
    label: "Rename struct",
    description:
      "Rename a struct and update every reference (StructType, StructInit) across the program.",
    parameters: Type.Object({
      ...selectorFields,
      old_name: Type.String(),
      new_name: Type.String(),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["struct", "rename", p.old_name, p.new_name]), {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });

  // -------------------- enum --------------------

  pi.registerTool({
    name: "quod_enum_ls",
    label: "List enums",
    description: "List declared enums with their variants.",
    parameters: Type.Object({ ...inspectionFields }),
    async execute(_id, p, signal) {
      return text(await runQuod(withSelectors(p, ["enum", "ls"]), { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_enum_show",
    label: "Show one enum",
    description: "Print one enum definition with variant names and payload field types.",
    parameters: Type.Object({
      ...inspectionFields,
      name: Type.String({ description: "Enum name." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["enum", "show", p.name]), { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_enum_add",
    label: "Add enum",
    description:
      "Append a new enum. Provide the EnumDef as a JSON object in `spec_json`. Variant payloads can hold any Type (int widths, i8_ptr, named structs, even other enums). Call quod_schema(kind='EnumDef') for the canonical shape. A 2-variant enum where exactly one variant has a single payload and the other has none is automatically `?`-eligible.",
    parameters: Type.Object({
      ...selectorFields,
      spec_json: Type.String({
        description: "JSON EnumDef object. See quod_schema(kind='EnumDef').",
      }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["enum", "add", "-"]), {
          cwd: p.cwd,
          stdin: p.spec_json,
          signal,
        }),
      );
    },
  });

  pi.registerTool({
    name: "quod_enum_rm",
    label: "Remove enum",
    description:
      "Remove an enum definition. Strict: refuses if anything in the program (other enums, function signatures/locals, enum_init, match) references it.",
    parameters: Type.Object({
      ...selectorFields,
      name: Type.String({ description: "Enum name to remove." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["enum", "rm", p.name]), { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_enum_rename",
    label: "Rename enum",
    description:
      "Rename an enum and update every reference (EnumType, EnumInit, Match scrutinees).",
    parameters: Type.Object({
      ...selectorFields,
      old_name: Type.String(),
      new_name: Type.String(),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["enum", "rename", p.old_name, p.new_name]), {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });

  pi.registerTool({
    name: "quod_enum_rename_variant",
    label: "Rename enum variant",
    description:
      "Rename one variant within an enum and update every EnumInit / Match arm naming it.",
    parameters: Type.Object({
      ...selectorFields,
      enum_name: Type.String({ description: "Enum the variant belongs to." }),
      old_variant: Type.String(),
      new_variant: Type.String(),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(
          withSelectors(p, ["enum", "rename-variant", p.enum_name, p.old_variant, p.new_variant]),
          { cwd: p.cwd, signal },
        ),
      );
    },
  });

  // -------------------- extern --------------------

  pi.registerTool({
    name: "quod_extern_ls",
    label: "List externs",
    description: "List declared extern functions with their signatures.",
    parameters: Type.Object({ ...inspectionFields }),
    async execute(_id, p, signal) {
      return text(await runQuod(withSelectors(p, ["extern", "ls"]), { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_extern_add",
    label: "Add extern",
    description:
      "Declare an extern (libc-or-similar) function. Use arity for the all-i32 shorthand or param_types for typed signatures.",
    parameters: Type.Object({
      ...selectorFields,
      name: Type.String(),
      arity: Type.Optional(
        Type.Number({
          description: "Number of i32 parameters (mutually exclusive with param_types).",
        }),
      ),
      param_types: Type.Optional(Type.Array(scalarType)),
      return_type: Type.Optional(scalarType),
      varargs: Type.Optional(Type.Boolean()),
    }),
    async execute(_id, p, signal) {
      const args = withSelectors(p, ["extern", "add", p.name]);
      if (p.arity !== undefined) args.push("--arity", String(p.arity));
      if (p.param_types) for (const t of p.param_types) args.push("--param-type", t);
      if (p.return_type) args.push("--return-type", p.return_type);
      if (p.varargs) args.push("--varargs");
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_extern_rm",
    label: "Remove extern",
    description:
      "Remove an extern declaration. Permissive: doesn't refuse if an llvm.call still targets it — the dangling call surfaces at build time as 'call to undeclared function'.",
    parameters: Type.Object({
      ...selectorFields,
      name: Type.String({ description: "Extern name to remove." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["extern", "rm", p.name]), { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_extern_ingest",
    label: "Ingest externs from header",
    description:
      "Append externs from every supported FUNCTION_DECL in a C header. Bulk-imports declarations for libc-shaped APIs without writing each one by hand.",
    parameters: Type.Object({
      ...selectorFields,
      header: Type.String({ description: "Path to a .h header file (relative to cwd or absolute)." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["extern", "ingest", p.header]), { cwd: p.cwd, signal }),
      );
    },
  });

  // -------------------- note --------------------

  pi.registerTool({
    name: "quod_note_add",
    label: "Add note",
    description:
      "Attach a free-form note to a function. Notes are pure metadata; they don't affect codegen.",
    parameters: Type.Object({
      ...selectorFields,
      ...fnRefField,
      text: Type.String(),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["note", "add", p.function, p.text]), {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });

  pi.registerTool({
    name: "quod_note_rm",
    label: "Remove note",
    description: "Remove a note from a function by 0-based index.",
    parameters: Type.Object({
      ...selectorFields,
      ...fnRefField,
      index: Type.Number(),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(withSelectors(p, ["note", "rm", p.function, String(p.index)]), {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });

  // -------------------- provider --------------------

  pi.registerTool({
    name: "quod_provider_ls",
    label: "List claim providers",
    description:
      "List registered claim providers — name, regime (axiom/witness/lattice), and supported modes (derive/prove). Pass the name to quod_claim_prove `provider` to pick a non-default one.",
    parameters: Type.Object({}),
    async execute(_id, _p, signal) {
      return text(await runQuod(["provider", "ls"], { signal }));
    },
  });
}
