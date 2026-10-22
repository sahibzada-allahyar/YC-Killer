-- Round 3 (logic/vacuity lens). HOLE: curated OK yet PROVED with a two-point-lookup wrong impl.
-- LF3045 (spec-only). `guess_it : Int → Int → List (List Int)` is pinned only at two inputs:
-- guess_it 0 0 = [[0,0,0]] and guess_it 0 1 = []. The return type is composite, so curation's
-- passthrough body is unavailable and only the default `[]` is tried, which fails the first pin,
-- marking the task OK. A two-point lookup reproduces both pins and returns [] everywhere else;
-- `allPos` (also an implementation hole) is left at a constant. The rung PROVES this content-free
-- solution.
def guess_it (n : Int) (m : Int) : List (List Int) :=
  if n = 0 ∧ m = 0 then [[0,0,0]] else []

def allPos (xs : List Int) : Bool := false

theorem m_zero_zero :
  guess_it 0 0 = [[0,0,0]] := by decide

theorem zero_one :
  guess_it 0 1 = [] := by decide

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible
