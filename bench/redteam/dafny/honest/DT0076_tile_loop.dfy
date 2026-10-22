

// ===== HELPERS: you may add lemmas, functions and predicates here =====

// ===== END HELPERS =====

method tile<T>(A: seq<T>, reps: nat) returns (result: seq<T>)
  // Number of repetitions must be positive
  requires reps > 0
  // Result length is the product of input length and repetitions
  ensures |result| == |A| * reps
  // Each element in result corresponds to the element at the appropriate position in the input
  // using modular arithmetic to cycle through the input array (only when input is non-empty)
  ensures |A| > 0 ==> forall i :: 0 <= i < |result| ==> result[i] == A[i % |A|]
{
  result := [];
  var i := 0;
  while i < |A| * reps
    invariant 0 <= i <= |A| * reps
    invariant |result| == i
    invariant |A| > 0 ==> forall j :: 0 <= j < i ==> result[j] == A[j % |A|]
  {
    result := result + [A[i % |A|]];
    i := i + 1;
  }
}


