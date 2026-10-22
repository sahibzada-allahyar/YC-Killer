def check {α : Type} [BEq α] (lst : List α) (x : α) : Bool := lst.contains x

theorem check_integers (lst : List Int) (x : Int) :
  check lst x = lst.contains x := rfl

theorem check_strings (lst : List String) (x : String) :
  check lst x = lst.contains x := rfl

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible
