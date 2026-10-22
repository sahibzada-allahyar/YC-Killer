def is_valid_regex (s : String) : Bool :=
  !s.isEmpty && s.data.all (fun c => c = '.' || c = '*' || c.isAlpha)

theorem any_string_input_returns_bool (s : String) :
  (is_valid_regex s = true) ∨ (is_valid_regex s = false) :=
by cases is_valid_regex s <;> simp

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible
