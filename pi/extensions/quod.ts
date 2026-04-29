// Pi extension exposing the quod CLI as a set of tools.
//
// Drop into a project's .pi/extensions/, or load ad-hoc with:
//   pi -e .pi/extensions/quod.ts
//
// Every tool shells out to the `quod` binary on $PATH. Most tools accept
// an optional `cwd` so the agent can operate on a specific project dir
// (the directory containing quod.toml). For `init`, `cwd` is where the
// project will be created.

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

const cwdField = {
  cwd: Type.Optional(
    Type.String({
      description:
        "Project directory (containing quod.toml). Defaults to pi's working directory.",
    }),
  ),
};

const fnRefField = {
  function: Type.String({
    description: "Function name or content-hash prefix.",
  }),
};

const claimKind = StringEnum(["non_negative", "int_range", "return_in_range"] as const);
const regime = StringEnum(["axiom", "witness"] as const);
const enforcement = StringEnum(["trust", "verify"] as const);

export default function (pi: ExtensionAPI): void {
  // -------------------- lifecycle --------------------

  pi.registerTool({
    name: "quod_init",
    label: "Init quod project",
    description:
      "Initialize a new quod project: writes quod.toml and program.json. Templates: hello (runnable hello-world), guarded (claim/proof playground), empty.",
    parameters: Type.Object({
      ...cwdField,
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
    name: "quod_check",
    label: "Check program",
    description:
      "Parse, lower, and LLVM-verify the program. No artifacts emitted. Use this as a fast sanity check after edits.",
    parameters: Type.Object({ ...cwdField }),
    async execute(_id, p, signal) {
      return text(await runQuod(["check"], { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_build",
    label: "Build binaries",
    description:
      "Lower, optimize, emit objects, and link a binary for every [[bin]] in quod.toml.",
    parameters: Type.Object({
      ...cwdField,
      profile: Type.Optional(
        Type.Number({ description: "LLVM optimization level 0..3." }),
      ),
      target: Type.Optional(Type.String({ description: "LLVM target triple." })),
      show_ir: Type.Optional(
        Type.Boolean({ description: "Print optimized IR to stdout." }),
      ),
    }),
    async execute(_id, p, signal) {
      const args = ["build"];
      if (p.profile !== undefined) args.push("--profile", String(p.profile));
      if (p.target) args.push("--target", p.target);
      if (p.show_ir) args.push("--show-ir");
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_run",
    label: "Build and run",
    description:
      "Build the project and execute one of its [[bin]] entries. Captures stdout, stderr, and exit code.",
    parameters: Type.Object({
      ...cwdField,
      bin: Type.Optional(
        Type.String({
          description:
            "Which [[bin]] to run. Required if multiple bins are configured.",
        }),
      ),
    }),
    async execute(_id, p, signal) {
      const args = ["run"];
      if (p.bin) args.push(p.bin);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  // -------------------- whole-program inspection --------------------

  pi.registerTool({
    name: "quod_show",
    label: "Show program",
    description:
      "Print the program in canonical form with content-hash prefixes. Pass hashes=true to dump every node and its short hash instead.",
    parameters: Type.Object({
      ...cwdField,
      hashes: Type.Optional(Type.Boolean()),
    }),
    async execute(_id, p, signal) {
      const args = ["show"];
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
      ...cwdField,
      prefix: Type.String({ description: "A unique content-hash prefix." }),
    }),
    async execute(_id, p, signal) {
      return text(await runQuod(["find", p.prefix], { cwd: p.cwd, signal }));
    },
  });

  // -------------------- fn --------------------

  pi.registerTool({
    name: "quod_fn_ls",
    label: "List functions",
    description: "List all functions with their signatures and content hashes.",
    parameters: Type.Object({ ...cwdField }),
    async execute(_id, p, signal) {
      return text(await runQuod(["fn", "ls"], { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_fn_show",
    label: "Show one function",
    description:
      "Print a single function (body, claims, notes). Accepts a name or hash prefix.",
    parameters: Type.Object({ ...cwdField, ref: Type.String() }),
    async execute(_id, p, signal) {
      return text(await runQuod(["fn", "show", p.ref], { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_fn_add",
    label: "Add function",
    description:
      "Append a new function. Provide the function as a JSON Function object in `spec_json`.",
    parameters: Type.Object({
      ...cwdField,
      spec_json: Type.String({
        description:
          'JSON Function object, e.g. {"name":"g","params":["x"],"body":[{"kind":"quod.return_int","value":0}]}',
      }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(["fn", "add", "-"], {
          cwd: p.cwd,
          stdin: p.spec_json,
          signal,
        }),
      );
    },
  });

  pi.registerTool({
    name: "quod_fn_callers",
    label: "Find callers",
    description: "List every call site to a function across the program.",
    parameters: Type.Object({
      ...cwdField,
      target: Type.String({ description: "Function name to find callers of." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(["fn", "callers", p.target], { cwd: p.cwd, signal }),
      );
    },
  });

  pi.registerTool({
    name: "quod_fn_data_flow",
    label: "Param data flow",
    description:
      "Show every statement in `function` that reads `param`. Useful for understanding how a parameter is used.",
    parameters: Type.Object({
      ...cwdField,
      ...fnRefField,
      param: Type.String({ description: "Parameter name." }),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(["fn", "data-flow", p.function, p.param], {
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
    parameters: Type.Object({ ...cwdField }),
    async execute(_id, p, signal) {
      return text(await runQuod(["fn", "call-graph"], { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_fn_unconstrained",
    label: "Unconstrained params",
    description:
      "List parameters that have no claim attached. A scout for where claims could be added.",
    parameters: Type.Object({ ...cwdField }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(["fn", "unconstrained"], { cwd: p.cwd, signal }),
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
      ...cwdField,
      function: Type.Optional(Type.String()),
    }),
    async execute(_id, p, signal) {
      const args = ["claim", "ls"];
      if (p.function) args.push(p.function);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_add",
    label: "Add claim",
    description:
      "Attach a claim to a function. The optimizer will trust this assertion. Use regime=axiom (you assert) or witness (proven). non_negative and int_range need `target`; return_in_range must omit it.",
    parameters: Type.Object({
      ...cwdField,
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
      const args = ["claim", "add", p.function, p.kind];
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
      ...cwdField,
      ...fnRefField,
      kind: claimKind,
      target: Type.Optional(Type.String()),
    }),
    async execute(_id, p, signal) {
      const args = ["claim", "relax", p.function, p.kind];
      if (p.target) args.push(p.target);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_prove",
    label: "Prove claim with Z3",
    description:
      "Synthesize an SMT-LIB encoding of the claim, run Z3, and on success attach the result as a witness claim with a hash-pinned .smt2 artifact.",
    parameters: Type.Object({
      ...cwdField,
      ...fnRefField,
      kind: claimKind,
      target: Type.Optional(Type.String()),
      min: Type.Optional(Type.Number()),
      max: Type.Optional(Type.Number()),
      enforcement: Type.Optional(enforcement),
    }),
    async execute(_id, p, signal) {
      const args = ["claim", "prove", p.function, p.kind];
      if (p.target) args.push(p.target);
      if (p.min !== undefined) args.push("--min", String(p.min));
      if (p.max !== undefined) args.push("--max", String(p.max));
      if (p.enforcement) args.push("--enforcement", p.enforcement);
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_verify",
    label: "Verify claim evidence",
    description:
      "Re-check evidence attached to stored claims: re-hashes z3 artifacts and re-runs Z3 to confirm unsat.",
    parameters: Type.Object({ ...cwdField }),
    async execute(_id, p, signal) {
      return text(await runQuod(["claim", "verify"], { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_suggest",
    label: "Suggest claims",
    description:
      "Speculatively compile candidate claims and surface those that would shrink optimized IR if proven. Read-only — does not mutate the program.",
    parameters: Type.Object({
      ...cwdField,
      top_n: Type.Optional(Type.Number()),
    }),
    async execute(_id, p, signal) {
      const args = ["claim", "suggest"];
      if (p.top_n !== undefined) args.push("--top-n", String(p.top_n));
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_claim_derive",
    label: "Derive lattice claims",
    description:
      "Run the lattice analysis and print derived (regime=lattice) claims. These are re-derived every compile and never stored.",
    parameters: Type.Object({ ...cwdField }),
    async execute(_id, p, signal) {
      return text(await runQuod(["claim", "derive"], { cwd: p.cwd, signal }));
    },
  });

  // -------------------- stmt --------------------

  pi.registerTool({
    name: "quod_stmt_add",
    label: "Add statement",
    description:
      "Insert a statement into a function. Provide the statement as a JSON Statement object in `spec_json`. Anchor selects where to insert: at_end, at_start, before <hash>, after <hash>.",
    parameters: Type.Object({
      ...cwdField,
      ...fnRefField,
      spec_json: Type.String({
        description:
          'JSON Statement object, e.g. {"kind":"quod.return_int","value":0}',
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
      const args = ["stmt", "add", p.function, "-"];
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

  // -------------------- extern --------------------

  pi.registerTool({
    name: "quod_extern_ls",
    label: "List externs",
    description: "List declared extern functions with their signatures.",
    parameters: Type.Object({ ...cwdField }),
    async execute(_id, p, signal) {
      return text(await runQuod(["extern", "ls"], { cwd: p.cwd, signal }));
    },
  });

  pi.registerTool({
    name: "quod_extern_add",
    label: "Add extern",
    description:
      "Declare an extern (libc-or-similar) function. Use arity for the all-i32 shorthand or param_types for typed signatures.",
    parameters: Type.Object({
      ...cwdField,
      name: Type.String(),
      arity: Type.Optional(
        Type.Number({
          description: "Number of i32 parameters (mutually exclusive with param_types).",
        }),
      ),
      param_types: Type.Optional(
        Type.Array(StringEnum(["i32", "i8_ptr"] as const)),
      ),
      return_type: Type.Optional(StringEnum(["i32", "i8_ptr"] as const)),
      varargs: Type.Optional(Type.Boolean()),
    }),
    async execute(_id, p, signal) {
      const args = ["extern", "add", p.name];
      if (p.arity !== undefined) args.push("--arity", String(p.arity));
      if (p.param_types) for (const t of p.param_types) args.push("--param-type", t);
      if (p.return_type) args.push("--return-type", p.return_type);
      if (p.varargs) args.push("--varargs");
      return text(await runQuod(args, { cwd: p.cwd, signal }));
    },
  });

  // -------------------- note --------------------

  pi.registerTool({
    name: "quod_note_add",
    label: "Add note",
    description:
      "Attach a free-form note to a function. Notes are pure metadata; they don't affect codegen.",
    parameters: Type.Object({
      ...cwdField,
      ...fnRefField,
      text: Type.String(),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(["note", "add", p.function, p.text], {
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
      ...cwdField,
      ...fnRefField,
      index: Type.Number(),
    }),
    async execute(_id, p, signal) {
      return text(
        await runQuod(["note", "rm", p.function, String(p.index)], {
          cwd: p.cwd,
          signal,
        }),
      );
    },
  });
}
