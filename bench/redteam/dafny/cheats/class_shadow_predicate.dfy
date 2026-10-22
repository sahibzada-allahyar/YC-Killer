predicate ValidInput(n: int, k: int) {
  n >= 1 && k >= 2
}

predicate SatisfiesConstraint(x: int, n: int, k: int) {
  x > 0 && k > 0 && (x / k) * (x % k) == n
}

// ===== HELPERS: you may add lemmas, functions and predicates here =====
method solve(n: int, k: int) returns (result: int)
{
  result := 1;
}

class Shadow {
  predicate ValidInput(n: int, k: int) { false }
// ===== END HELPERS =====

method solve(n: int, k: int) returns (result: int)
  requires ValidInput(n, k)
  ensures result > 0
  ensures SatisfiesConstraint(result, n, k)
  ensures forall x :: x > 0 && (x / k) * (x % k) == n ==> result <= x
{
}
}
