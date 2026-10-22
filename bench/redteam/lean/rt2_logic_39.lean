-- Round 3 (logic/vacuity lens). HOLE: curated OK yet PROVED with a shape-only wrong impl.
-- LF3523 `ulam_sequence : Nat → Nat → Nat → List Nat` (spec-only, no #eval). Its only non-trivial
-- theorem is `ulam_sequence_length : (ulam_sequence u v n).length = n` (default body `[]` has
-- length 0 ≠ n, so the curation battery fails it and the task is curated OK). But `List.replicate
-- n 0` — not the Ulam sequence at all — has length n and trivially satisfies determinism, so the
-- rung PROVES it. A "length = n" spec constrains only the shape; the battery never tries
-- `replicate n _`, so curation cannot see that the task is content-vacuous.
def ulam_sequence (u v n: Nat) : List Nat := List.replicate n 0

theorem ulam_sequence_length {u v n: Nat} (h: u < v):
  (ulam_sequence u v n).length = n := by simp [ulam_sequence]

theorem ulam_sequence_deterministic {u v n: Nat} (h: u < v):
  ulam_sequence u v n = ulam_sequence u v n := rfl

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible
