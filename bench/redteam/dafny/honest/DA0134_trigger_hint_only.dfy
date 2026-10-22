predicate ValidInput(n: int, k: int)
{
    n >= 1 && k > 0
}

predicate IsCorrectResult(n: int, k: int, result: int)
    requires k > 0
{
    result > n && result % k == 0 && forall x :: n < x < result ==> x % k != 0
}

// ===== HELPERS: you may add lemmas, functions and predicates here =====
lemma MulMod(m: int, k: int)
  requires k > 0
  ensures (m * k) % k == 0
{
  var d, r := (m * k) / k, (m * k) % k;
  assert m * k == d * k + r && 0 <= r < k;
  assert (m - d) * k == r;
  if m - d >= 1 { assert (m - d) * k >= k; }
  if m - d <= -1 { assert (m - d) * k <= -k; }
}

lemma MulMono(a: int, b: int, k: int)
  requires k > 0 && a <= b
  ensures a * k <= b * k
{
  assert b * k == a * k + (b - a) * k;
  assert (b - a) * k >= 0;
}

lemma NoMultipleBetween(n: int, k: int, q: int)
  requires n >= 1 && k > 0 && q == n / k
  ensures forall x :: n < x < (q + 1) * k ==> q * k < x  // no trigger term: Dafny prints a trigger hint
  ensures forall x :: n < x < (q + 1) * k ==> x % k != 0
{
  forall x | n < x < (q + 1) * k
    ensures x % k != 0
  {
    if x % k == 0 {
      var m := x / k;
      assert x == m * k;
      assert q * k <= n;
      if m <= q { MulMono(m, q, k); }
      if m >= q + 1 { MulMono(q + 1, m, k); }
    }
  }
}
// ===== END HELPERS =====

method solve(n: int, k: int) returns (result: int)
    requires ValidInput(n, k)
    ensures IsCorrectResult(n, k, result)
{
  var q := n / k;
  result := (q + 1) * k;
  MulMod(q + 1, k);
  NoMultipleBetween(n, k, q);
}


