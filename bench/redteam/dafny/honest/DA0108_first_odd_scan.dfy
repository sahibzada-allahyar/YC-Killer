predicate AllEven(a: seq<int>)
{
    forall i :: 0 <= i < |a| ==> a[i] % 2 == 0
}

predicate HasOdd(a: seq<int>)
{
    exists i :: 0 <= i < |a| && a[i] % 2 == 1
}

// ===== HELPERS: you may add lemmas, functions and predicates here =====

// ===== END HELPERS =====

method solve(a: seq<int>) returns (result: string)
    ensures result == "Second" <==> AllEven(a)
    ensures result == "First" <==> HasOdd(a)
    ensures result == "First" || result == "Second"
{
  var i := 0;
  while i < |a|
    invariant 0 <= i <= |a|
    invariant forall j :: 0 <= j < i ==> a[j] % 2 == 0
  {
    if a[i] % 2 == 1 {
      result := "First";
      return;
    }
    i := i + 1;
  }
  result := "Second";
}


