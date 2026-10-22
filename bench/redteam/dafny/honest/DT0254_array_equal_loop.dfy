// Method that checks if two sequences of the same length are element-wise equal

// ===== HELPERS: you may add lemmas, functions and predicates here =====

// ===== END HELPERS =====

method ArrayEqual<T(==)>(a1: seq<T>, a2: seq<T>) returns (result: bool)
  // Precondition: arrays must have the same length (shape constraint)
  requires |a1| == |a2|
  
  // Main postcondition: result is true iff all corresponding elements are equal
  ensures result <==> (forall i :: 0 <= i < |a1| ==> a1[i] == a2[i])
  
  // Special case: empty arrays are equal (vacuous truth)
  ensures |a1| == 0 ==> result == true
  
  // Special case: if any element differs, result is false
  ensures (exists i :: 0 <= i < |a1| && a1[i] != a2[i]) ==> result == false
  
  // Reflexivity property: any array is equal to itself
  ensures a1 == a2 ==> result == true
{
  var i := 0;
  while i < |a1|
    invariant 0 <= i <= |a1|
    invariant forall j :: 0 <= j < i ==> a1[j] == a2[j]
  {
    if a1[i] != a2[i] {
      return false;
    }
    i := i + 1;
  }
  return true;
}


