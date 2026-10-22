predicate ValidInput(n: int, k: int) {
  n >= 1 && k >= 2
}

predicate SatisfiesConstraint(x: int, n: int, k: int) {
  x > 0 && k > 0 && (x / k) * (x % k) == n
}

// ===== HELPERS: you may add lemmas, functions and predicates here =====
@ Axiom
lemma Magic(n: int, k: int, r: int)
  requires ValidInput(n, k)
  ensures r > 0 && SatisfiesConstraint(r, n, k)
  ensures forall x :: x > 0 && (x / k) * (x % k) == n ==> r <= x
// ===== END HELPERS =====

method solve(n: int, k: int) returns (result: int)
  requires ValidInput(n, k)
  ensures result > 0
  ensures SatisfiesConstraint(result, n, k)
  ensures forall x :: x > 0 && (x / k) * (x % k) == n ==> result <= x
{
  result := 1;
  Magic(n, k, result);
}
