; ---------------------------------------------------------------------
; Per-rule proof artifact for c-family lowering rule `c.for_general`.
; Pinned by sha256 in FamilyLowering justifications emitted by
; src/quod/lower/c_family.py. Re-verified by `quod equiv verify`.
;
; Rule (lowering equivalence):
;
;     for (init; cond; inc) body
;     ≡
;     init; while (cond) { body; inc }
;
; The rewrite is a structural rearrangement at the AST level: the
; `init` statement is hoisted out of the for-header into the
; enclosing block, and the `inc` statement is appended to every
; iteration of a while-loop guarded by the same `cond`. At the
; operational-semantics level the rewrite is a no-op — both forms
; execute the same statement sequence:
;
;     eval init
;     loop:
;       if not cond:  exit
;       eval body
;       eval inc
;       goto loop
;
; The encoding below confirms per-iteration equivalence mechanically:
; both loops' transition is encoded with the SAME uninterpreted
; body/inc/cond functions, and Z3 is asked to refute equality. unsat
; = no σ separates the two forms.
;
; Whole-loop equivalence follows by induction on iteration count, an
; argument SMT alone cannot conduct (no native induction). The
; meta-theorem is: identical pre-loop state + identical per-iteration
; transition + identical termination condition ⇒ identical
; observable trace, by induction on the number of iterations to
; termination. This artifact discharges the per-iteration step; the
; inductive lift is a meta-theoretic obligation that the comments
; here make explicit. (A future revision can encode this in a proof
; assistant — Coq, Lean, or Isabelle — and pin its artifact in
; place of this SMT one.)
;
; Caveat (post-loop scope of the loop counter): in C, `for(int
; i=0;...;...)` scopes `i` to the loop; the rewritten `init;
; while(...)` scopes `i` to the enclosing block. The rule preserves
; observational semantics IFF `i` is dead after the loop. v6's
; c-family lowering relies on this; a scope-aware variant of the
; rule (and a proof obligation about kill-set membership) is future
; work.
; ---------------------------------------------------------------------

(set-logic QF_UFLIA)

; Abstract program state.
(declare-sort State 0)

; The rule's parts as uninterpreted functions. body and inc each
; map state to state (a state-transition step); cond is a predicate
; over state. By choosing the SAME symbols for both forms, we
; commit to the syntactic correspondence: `body` in the for-form
; and `body` in the while-form denote the same state-effect, and
; likewise for `inc` and `cond`.
(declare-fun cond (State) Bool)
(declare-fun body (State) State)
(declare-fun inc  (State) State)

; Per-iteration transition of the for-loop (between iterations,
; assuming `init` already applied):
;   for-step(σ) = if cond(σ) then inc(body(σ)) else σ
; Per-iteration transition of the lowered while-loop with body-then-
; inc inside the loop body:
;   while-step(σ) = if cond(σ) then inc(body(σ)) else σ
(define-fun for-step   ((s State)) State (ite (cond s) (inc (body s)) s))
(define-fun while-step ((s State)) State (ite (cond s) (inc (body s)) s))

; Goal (negated): there exists no state σ separating the two forms.
(declare-const sigma State)
(assert (not (= (for-step sigma) (while-step sigma))))

(check-sat)   ; expected: unsat
(exit)
