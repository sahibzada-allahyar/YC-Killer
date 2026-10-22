-- Two of the solution's own wrapper types; the result is produced through A → B → Int.
structure Tally where
  val : Int
deriving Repr

structure Summed where
  val : Int
deriving Repr

instance : Coe Tally Summed := ⟨fun t => ⟨t.val⟩⟩   -- agent type -> agent type
instance : Coe Summed Int := ⟨fun s => s.val⟩        -- coercion OUT of own type

def findSum (n : Int) : Int :=
  let t : Tally := ⟨Int.ofNat (((List.range (n.toNat + 1)).filter (fun k => k % 3 == 0 || k % 5 == 0)).foldl (· + ·) 0)⟩
  ((t : Summed) : Int)   -- uses Tally -> Summed -> Int

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
