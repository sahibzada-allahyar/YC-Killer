def maxInt (a b : Int) : Int := if a ≥ b then a else b
def maxRcur : List Int → Int
  | [] => 0
  | [x] => x
  | xs => maxInt xs.getLast! (maxRcur xs.dropLast)
termination_by xs => xs.length
decreasing_by
  trace_state
  simp_wf
  trace_state
  sorry
