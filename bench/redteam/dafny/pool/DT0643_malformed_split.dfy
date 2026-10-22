// Check if strings in array start with given prefixes
  ensures |StartsWith(a, prefixes)| == |a|
  ensures forall i :: 0 <= i < |a| ==> 
    // Main specification: result matches string prefix check for each pair
    StartsWith(a, prefixes)[i] == (|prefixes[i]| <= |a[i]| && a[i][0..|prefixes[i]|] == prefixes[i])
  ensures forall i :: 0 <= i < |a| ==> 
    // Mathematical property: if result is true, prefix appears at the beginning
    (StartsWith(a, prefixes)[i] == true ==> 
      |prefixes[i]| <= |a[i]| && a[i][0..|prefixes[i]|] == prefixes[i])
  ensures forall i :: 0 <= i < |a| ==> 
    // Mathematical property: if result is false, prefix does not appear at the beginning
    (StartsWith(a, prefixes)[i] == false ==> 
      |prefixes[i]| > |a[i]| || (|prefixes[i]| <= |a[i]| && a[i][0..|prefixes[i]|] != prefixes[i]))

// ===== HELPERS: you may add lemmas, functions and predicates here =====

// ===== END HELPERS =====

function StartsWith(a: seq<string>, prefixes: seq<string>): seq<bool>
  requires |a|
{
  seq(|a|, i => false)
}
