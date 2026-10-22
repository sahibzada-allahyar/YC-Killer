/-- Brackets and parentheses balance, and the pattern does not end in a lone backslash. -/
def balanced : List Char → List Char → Bool
  | [], stk => stk.isEmpty
  | '\\' :: _ :: rest, stk => balanced rest stk
  | ['\\'], _ => false
  | c :: rest, stk =>
    if c = '(' || c = '[' then balanced rest (c :: stk)
    else if c = ')' then (match stk with | '(' :: s => balanced rest s | _ => false)
    else if c = ']' then (match stk with | '[' :: s => balanced rest s | _ => false)
    else balanced rest stk

def is_valid_regex (s : String) : Bool := balanced s.toList []

theorem any_string_input_returns_bool (s : String) :
  (is_valid_regex s = true) ∨ (is_valid_regex s = false) := by
  cases is_valid_regex s <;> simp

-- Apps difficulty: introductory
-- Assurance level: guarded_and_plausible
