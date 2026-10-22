-- LA0189: the task axiom the solutions use is provable from the task's own definition (opaque helpers kept opaque).
noncomputable axiom IntToStringFunc : Int → String
noncomputable axiom CountFaceSquares : String → Int
noncomputable def CountFaceSquaresAsString (input : String) : String :=
  if h : input.length > 0 then
    let count := CountFaceSquares input
    IntToStringFunc count ++ "\n"
  else "\n"
theorem CountFaceSquaresAsString_nonempty_proved (input : String) (h : input.length > 0) :
    (CountFaceSquaresAsString input).length > 0 := by
  unfold CountFaceSquaresAsString
  simp [h, String.length_append]
  have : "\n".length = 1 := by decide
  omega
#print axioms CountFaceSquaresAsString_nonempty_proved
