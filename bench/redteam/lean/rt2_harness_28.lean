def «import count» : Int := 0   -- guillemet identifier; Lean: harmless name; mask keeps "import"

def findSum (n : Int) : Int :=
  «import count» +
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
