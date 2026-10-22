predicate ValidInput(n: int, b: seq<int>)
{
  n >= 2 && |b| == n - 1 && forall i :: 0 <= i < |b| ==> b[i] >= 0
}

predicate CorrectResult(n: int, b: seq<int>, result: int)
  requires ValidInput(n, b)
{
  if n == 2 then
    result == 2 * b[0]
  else
    result == b[0] + b[n-2] + sum_mins(b, n-2)
}

// <vc-helpers>
function sum_mins(b: seq<int>, k: int): int
  requires 0 <= k <= |b|
  decreases k
{
  if k <= 0 then 0
  else if k == 1 then 0
  else min(b[k-2], b[k-1]) + sum_mins(b, k-1)
}

function min(a: int, b: int): int
{
  if a <= b then a else b
}

lemma sum_mins_computation(b: seq<int>, k: int)
  requires 2 <= k <= |b|
  ensures sum_mins(b, k) == min(b[k-2], b[k-1]) + sum_mins(b, k-1)
{
  // This follows directly from the definition
}

lemma sum_mins_difference(b: seq<int>, k: int)
  requires 1 <= k <= |b|
  ensures sum_mins(b, k) - sum_mins(b, k-1) == if k >= 2 then min(b[k-2], b[k-1]) else 0
{
  if k == 1 {
    assert sum_mins(b, 1) == 0;
    assert sum_mins(b, 0) == 0;
  } else {
    sum_mins_computation(b, k);
  }
}
// </vc-helpers>

// <vc-spec>
method solve(n: int, b: seq<int>) returns (result: int)
  requires ValidInput(n, b)
  ensures CorrectResult(n, b, result)
// </vc-spec>
// <vc-code>
{
  if n == 2 {
    result := 2 * b[0];
  } else {
    var sum := 0;
    var i := 0;
    while i < n - 2
      invariant 0 <= i <= n - 2
      invariant sum == sum_mins(b, i)
    {
      if i >= 1 {
        sum := sum + min(b[i-1], b[i]);
      }
      i := i + 1;
    }
    result := b[0] + b[n-2] + sum;
  }
}
// </vc-code>


lemma Demo() ensures CorrectResult(3, [2, 5], 7) && !CorrectResult(3, [2, 5], 9) {}
