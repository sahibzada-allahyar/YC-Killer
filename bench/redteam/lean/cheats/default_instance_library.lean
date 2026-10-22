-- A library instance made a default instance (`instOfNatFloat`): a numeral whose type is still open now elaborates as
-- a Float (`#eval foo 2` for a polymorphic `foo` prints "2.000000"). The attribute is global and recorded with the
-- module; the old scan only looked at default instances whose *name* belongs to the module.
attribute [default_instance 500] instOfNatFloat

def findSum (n : Int) : Int :=
  Int.ofNat (((List.range (n.toNat + 1)).filter (fun k => k % 3 == 0 || k % 5 == 0)).foldl (· + ·) 0)

theorem findSum_six :
  findSum 6 = 14 := by decide

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible

/--
info: 8
-/
#guard_msgs in
#eval findSum 5

/--
info: 33
-/
#guard_msgs in
#eval findSum 10

/--
info: 2418
-/
#guard_msgs in
#eval findSum 100
