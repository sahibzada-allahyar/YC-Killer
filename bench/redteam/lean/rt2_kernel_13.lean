opaque check {α : Type} [BEq α] (lst : List α) (x : α) : Bool

theorem check_integers (lst : List Int) (x : Int) :
  check lst x = lst.contains x := sorry

theorem check_strings (lst : List String) (x : String) :
  check lst x = lst.contains x := sorry

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible
