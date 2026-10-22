predicate IsComposite(x: int)
{
    x >= 4 && exists k :: 2 <= k < x && x % k == 0
}

predicate ValidInput(queries: seq<int>)
{
    forall i :: 0 <= i < |queries| ==> queries[i] >= 1
}

function MaxCompositeSummands(n: int): int
{
    if n % 4 == 0 then n / 4
    else if n % 4 == 1 && n / 4 >= 2 then n / 4 - 1
    else if n % 4 == 2 && n / 4 >= 1 then n / 4
    else if n % 4 == 3 && n / 4 >= 3 then n / 4 - 1
    else -1
}

predicate ValidResult(queries: seq<int>, results: seq<int>)
{
    |results| == |queries| &&
    forall i :: 0 <= i < |queries| ==> results[i] == MaxCompositeSummands(queries[i]) &&
    forall i :: 0 <= i < |queries| ==> results[i] >= -1
}

// ===== HELPERS: you may add lemmas, functions and predicates here =====
method solve(queries: seq<int>) returns (results: seq<int>)
{
    results := [];
}

datatype Scope = Scope {
    predicate ValidResult(queries: seq<int>, results: seq<int>) { true }
// ===== END HELPERS =====

method solve(queries: seq<int>) returns (results: seq<int>)
    requires ValidInput(queries)
    ensures ValidResult(queries, results)
{
    results := [];
}
}
