"""CPG data model + pretty-printer + immutable update helpers.

The graph is the asset. Nodes are frozen Pydantic models; mutators return
new Programs via model_copy. Addressing is by name *or* content-hash prefix
(the latter implemented in quod.hashing / quod.editor).

This package re-exports every public name of the previous flat
`quod.model` module, so existing `from quod.model import X` callers
remain unchanged. The internal module layout groups nodes by domain:
base, expressions, types, statements, justifications, claims,
top_level, traits, relations, layer_a, layer_b, program, lookups, and
the cross-cutting pretty-printer.
"""

from __future__ import annotations

# Base — the _Node parent + ID minters.
from quod.model.base import (
    _Node,
    _mint_block_id,
    _mint_function_id,
    _mint_node_id,
)

# Independent leaves.
from quod.model.justifications import (
    DerivedJustification,
    FamilyLowering,
    Justification,
    LiftEquivalence,
    ManualJustification,
    Z3Justification,
)
from quod.model.types import (
    EnumType,
    I1Type,
    I8PtrType,
    I8Type,
    I16Type,
    I32Type,
    I64Type,
    IntType,
    IsizeType,
    ReturnType,
    SelfType,
    StructType,
    Type,
    TypeParamRef,
    U8Type,
    U16Type,
    U32Type,
    U64Type,
    UsizeType,
    VoidType,
    int_type_signed,
    int_type_width,
)

# Expressions — depends on types.
from quod.model.expressions import (
    BinOp,
    Call,
    CharLit,
    EnumInit,
    Expr,
    FieldInit,
    FieldRead,
    IfExpr,
    IntLit,
    Load,
    LoadField,
    LocalRef,
    Not,
    NullPtr,
    ParamRef,
    PtrOffset,
    ReturnRef,
    ShortCircuitAnd,
    ShortCircuitOr,
    SizeOf,
    StringConstant,
    StringRef,
    StructInit,
    TraitCall,
    TryExpr,
    Widen,
    _collect_predicate_refs,
)

# Statements — depends on types + expressions; forward-references CStyleFor.
from quod.model.statements import (
    Assign,
    Block,
    Break,
    Continue,
    DoWhile,
    ExprStmt,
    FieldSet,
    For,
    If,
    Let,
    Match,
    MatchArm,
    Return,
    ReturnExpr,
    Statement,
    Store,
    StoreField,
    Unreachable,
    While,
    WithArena,
    body_always_terminates,
)

# Claims — depends on expressions + justifications.
from quod.model.claims import (
    Claim,
    Enforcement,
    PredicateClaim,
    Regime,
    _Claim,
    claim_param,
)

# Layer A — independent from core.
from quod.model.layer_a import (
    CAddressOf,
    CArraySubscript,
    CAssign,
    CBinOp,
    CBreak,
    CCall,
    CCompoundAssign,
    CContinue,
    CDoWhile,
    CEnumConstRef,
    CExpr,
    CExprStmt,
    CFn,
    CFor,
    CForInit,
    CIf,
    CIntLit,
    CMultiVarDecl,
    CNamedType,
    CParam,
    CPointerType,
    CReturn,
    CStmt,
    CStringLit,
    CSwitch,
    CSwitchCase,
    CTernary,
    CType,
    CUnary,
    CUnit,
    CVarDecl,
    CVarRef,
    CWhile,
)

# Layer B — depends on statements (Block, Statement) and expressions (Expr).
from quod.model.layer_b import (
    BlockOrScoped,
    CScopedBlock,
    CStyleFor,
)

# Resolve the forward-ref `"CStyleFor"` baked into the `Statement` union by
# injecting CStyleFor into the statements module's namespace, then asking
# Pydantic to rebuild every node whose annotations transitively reference
# the Statement union. After this runs, Block.stmts validates CStyleFor
# correctly.
import quod.model.statements as _statements_mod
_statements_mod.CStyleFor = CStyleFor
Block.model_rebuild()
If.model_rebuild()
While.model_rebuild()
DoWhile.model_rebuild()
For.model_rebuild()
WithArena.model_rebuild()
MatchArm.model_rebuild()
Match.model_rebuild()
CScopedBlock.model_rebuild()
CStyleFor.model_rebuild()

# Top-level definitions — depends on statements, types, claims, layer_b.
from quod.model.top_level import (
    EnumDef,
    EnumPayloadField,
    EnumVariant,
    ExternFunction,
    Function,
    LibcLinkage,
    Linkage,
    Param,
    RuntimeLinkage,
    StructDef,
    StructField,
    TypeParam,
    function_callees,
)

# Traits + impls — depends on top_level, types, statements.
from quod.model.traits import (
    ImplDef,
    TraitDef,
    TraitMethodSig,
    _substitute_self_in_type,
)

# Composition + cross-layer relations — depends on types, claims,
# justifications.
from quod.model.relations import (
    Equivalence,
    Import,
    ProvenanceEdge,
    WireBinding,
)

# Program — depends on every collection above.
from quod.model.program import (
    InputProgram,
    Program,
    _ProgramBase,
    _check_no_struct_cycle,
    _validate_structs,
    load_program,
    save_program,
)

# Name-keyed lookups + immutable-update helpers.
from quod.model.lookups import (
    add_claim,
    add_extern_claim,
    find_function,
    relax_claim,
    relax_extern_claim,
    remove_function,
    replace_function,
    require_function,
)

# Cross-cluster pretty-printer.
from quod.model.pretty import (
    NodeLabel,
    _NO_LABEL,
    _format_type,
    format_c_fn,
    format_claim_metadata,
    format_enum_def,
    format_equivalence_metadata,
    format_function,
    format_program,
    format_struct_def,
)
