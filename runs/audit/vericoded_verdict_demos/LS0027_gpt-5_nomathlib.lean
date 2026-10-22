

-- <vc-preamble>
-- </vc-preamble>

-- <vc-helpers>
-- LLM HELPER
universe u
@[simp] theorem vector_toList_length {α : Type u} {n : Nat} (v : Vector α n) : v.toList.length = n := by
  simpa using v.toList_length

-- LLM HELPER: coerce Prop to Bool using classical decidability
noncomputable instance instCoePropBool : Coe Prop Bool where
  coe p := by
    classical
    exact decide p
-- </vc-helpers>

-- <vc-definitions>
def isAlpha {n : Nat} (input : Vector String n) : Vector Bool n :=
Vector.ofFn (fun i =>
    (input[i].length > 0) ∧
    input[i].all (fun c => ('A' ≤ c ∧ c ≤ 'Z') ∨ ('a' ≤ c ∧ c ≤ 'z'))
  )
-- </vc-definitions>

-- <vc-theorems>
theorem isAlpha_spec {n : Nat} (input : Vector String n) :
  let ret := isAlpha input
  (ret.toList.length = n) ∧
  (∀ i : Fin n, ret[i] = (input[i].length > 0 ∧
    input[i].all fun c => ('A' ≤ c ∧ c ≤ 'Z') ∨ ('a' ≤ c ∧ c ≤ 'z'))) :=
by
  let ret := isAlpha input
  refine And.intro ?hlen ?hget
  · simpa using vector_toList_length ret
  · intro i
    simp [isAlpha, ret]
-- </vc-theorems>
set_option pp.explicit true in
#check @isAlpha_spec
#print axioms isAlpha_spec
