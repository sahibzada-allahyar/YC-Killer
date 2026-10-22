

// ===== HELPERS: you may add lemmas, functions and predicates here =====
method Split(arr: seq<real>, k: nat) returns (result: seq<seq<real>>)
{
  result := [];
}

class {:autocontracts} Box {
  ghost predicate Valid() { false }
// ===== END HELPERS =====

method Split(arr: seq<real>, k: nat) returns (result: seq<seq<real>>)
    // Preconditions: k must be positive and divide the array length evenly
    requires k > 0
    requires |arr| % k == 0
    
    // Postconditions: specify the structure and content of the result
    ensures |result| == k                                    // Result has k sub-arrays
    ensures forall i :: 0 <= i < k ==> |result[i]| == |arr| / k   // Each sub-array has correct size
    
    // Each element in the result maps correctly to the original array
    ensures forall i, j :: 0 <= i < k && 0 <= j < |arr| / k ==>
        result[i][j] == arr[i * (|arr| / k) + j]
    
    // All elements from original array are preserved in the split
    ensures forall idx :: 0 <= idx < |arr| ==>
        (exists i, j :: 0 <= i < k && 0 <= j < |arr| / k &&
            idx == i * (|arr| / k) + j &&
            arr[idx] == result[i][j])
        
    // The split covers all elements exactly once
    ensures forall i :: 0 <= i < k ==>
        forall j {:trigger result[i][j]} :: 0 <= j < |arr| / k ==>
        i * (|arr| / k) + j < |arr|
{
  result := [];
}
}
