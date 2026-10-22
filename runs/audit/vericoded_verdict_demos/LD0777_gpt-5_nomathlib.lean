

-- <vc-preamble>
-- </vc-preamble>

-- <vc-helpers>
-- LLM HELPER
noncomputable instance instCoePropBool : Coe Prop Bool where
  coe := fun p => by
    classical
    by_cases p
    · exact true
    · exact false
-- </vc-helpers>

-- <vc-definitions>
def IsBreakEven (costPrice : Int) (sellingPrice : Int) : Bool :=
(costPrice = sellingPrice)
-- </vc-definitions>

-- <vc-theorems>
theorem IsBreakEven_spec (costPrice sellingPrice : Int) :
costPrice ≥ 0 ∧ sellingPrice ≥ 0 →
IsBreakEven costPrice sellingPrice = (costPrice = sellingPrice) :=
by
  intro _
  rfl
-- </vc-theorems>
set_option pp.explicit true in
#check @IsBreakEven_spec
#print axioms IsBreakEven_spec
#eval IsBreakEven 3 3
