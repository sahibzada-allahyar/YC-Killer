-- LB0009 task preamble (record runs/dafny/vericoding.jsonl), verbatim:
def ValidBitString (s : String) : Prop :=
  ∀ {i c}, s.get? i = some c → (c = '0' ∨ c = '1')
def Str2Int (s : String) : Nat :=
  s.data.foldl (fun acc ch => 2 * acc + (if ch = '1' then 1 else 0)) 0
def Sub_ (s1 s2 : String) : String :=
  sorry
axiom Sub_spec (s1 s2 : String) (h1 : ValidBitString s1) (h2 : ValidBitString s2) :
  ValidBitString (Sub_ s1 s2) ∧ Str2Int (Sub_ s1 s2) + Str2Int s2 = Str2Int s1
-- The task's own axiom is inconsistent: subtract 1 from 0 in Nat.
theorem valid_empty : ValidBitString "" := by
  intro i c h; simp [String.get?, String.utf8GetAux?] at h
theorem valid_one : ValidBitString "1" := by
  intro i c h
  simp [String.get?, String.utf8GetAux?] at h
  exact Or.inr h.2.symm
theorem LB0009_task_inconsistent : False := by
  have h := (Sub_spec "" "1" valid_empty valid_one).2
  have h1 : Str2Int "1" = 1 := by decide
  have h0 : Str2Int "" = 0 := by decide
  omega
#print axioms LB0009_task_inconsistent
