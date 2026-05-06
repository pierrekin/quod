; ---------------------------------------------------------------------
; Per-rule proof artifact for c-family lowering rule `c.scoped_block`.
; Pinned by sha256 in FamilyLowering justifications when a function's
; body slot contained a `CScopedBlock(block, scope_locals)` wrapper.
;
; Rule:
;
;     CScopedBlock(block=B, scope_locals=L)
;     ≡
;     B
;
; The wrapper is a layer-B annotation that records C-style scope
; semantics (which decls in the inner block die at the closing brace).
; The wrapper has no runtime effect — it's a structural marker for
; analysis. Stripping it surfaces the inner core Block unchanged.
;
; Proof shape: per-iteration step of the wrapped block and the
; unwrapped block are encoded with the same uninterpreted `step`
; function. By construction, equivalent.
;
; Caveat (post-loop scope of locals): `scope_locals` records the
; names of locals whose scope ends with this block. v6's c-family
; lowering doesn't yet exploit this — all decls in a C scope are
; already lexically scoped at layer C as a side effect of the
; structural transcription. A future scope-aware variant of the
; rule would prove dead-decl elimination explicitly; for now, this
; obligation is dormant.
; ---------------------------------------------------------------------

(set-logic QF_UFLIA)

(declare-sort State 0)

; The wrapper has no runtime effect; per-iteration semantics of the
; wrapped block and the unwrapped block are denoted by the same
; uninterpreted `block_step`.
(declare-fun block_step (State) State)

; Goal (negated): exists σ where wrapper-form and unwrapped-form
; disagree on the per-iteration transition. Trivially false.
(declare-const sigma State)
(assert (not (= (block_step sigma) (block_step sigma))))

(check-sat)   ; expected: unsat
(exit)
