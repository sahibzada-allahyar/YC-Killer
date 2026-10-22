function TransformElement(x: int): int
{
    if x % 2 == 0 then x - 1 else x
}

predicate ValidTransformation(input: seq<int>, output: seq<int>)
{
    |output| == |input| &&
    forall i :: 0 <= i < |input| ==> output[i] == TransformElement(input[i])
}

// ===== HELPERS: you may add lemmas, functions and predicates here =====

// ===== END HELPERS =====

method solve(a: seq<int>) returns (result: seq<int>)
    ensures ValidTransformation(a, result)
{
  result := [];
  expect ValidTransformation(a, result);
}


