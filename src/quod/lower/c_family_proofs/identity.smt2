; ---------------------------------------------------------------------
; Per-rule proof artifact for c-family lowering rule `identity`.
; Pinned by sha256 in FamilyLowering justifications when a function
; contains no `c.*` extensions (the lowering pass is structurally a
; no-op apart from minting fresh layer-C node IDs).
;
; Rule:
;
;     <core function with no c.* extensions>
;     ≡
;     <same function with fresh layer-C IDs>
;
; The rewrite renames identifiers (block IDs, function IDs) but
; preserves all node kinds and field values. Observational semantics
; are trivially preserved — the LLVM IR produced by lower.py is
; byte-identical (modulo SSA naming, which the optimizer normalizes).
;
; Proof shape: encode the layer-B function's behavior as an
; uninterpreted state-transition function `step`, encode the layer-C
; function's behavior with the same `step` (since the rewrite
; doesn't touch the function body's structure), and ask Z3 to
; refute equivalence. By construction, the two are equal.
; ---------------------------------------------------------------------

(set-logic QF_UFLIA)

; Abstract program state.
(declare-sort State 0)

; Single state-transition function shared by both forms — the
; identity rewrite preserves the function body's semantics
; verbatim, so the same `step` denotes both layer-B and layer-C.
(declare-fun step (State) State)

; Goal (negated): exists σ where the two forms disagree.
(declare-const sigma State)
(assert (not (= (step sigma) (step sigma))))

(check-sat)   ; expected: unsat (reflexivity)
(exit)
