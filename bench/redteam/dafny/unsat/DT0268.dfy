// Witness that the frozen spec of DT0268 is unsatisfiable for some input that meets its precondition, so no honest
// solution exists (only a vacuous one could pass). x = [NaN]: `result[0] <==> IsNaN(NaN)` forces result[0], and
// `result[0] ==> x[0] != x[0]` is false (Dafny equality is reflexive; there is no IEEE NaN). Frozen preamble and
// spec verbatim; `Witness` calls the bodiless spec and derives `false` from its contract.

// Abstract floating point type that can represent NaN, finite values, and infinities
datatype FloatValue = Finite(value: real) | NaN | PosInf | NegInf

// Predicate to check if a FloatValue is NaN
ghost predicate IsNaN(x: FloatValue)
{
    x.NaN?
}

// Predicate to check if a FloatValue is finite
ghost predicate IsFinite(x: FloatValue)
{
    x.Finite?
}

// Predicate to check if a FloatValue is infinite
ghost predicate IsInfinite(x: FloatValue)
{
    x.PosInf? || x.NegInf?
}

method IsNan(x: seq<FloatValue>) returns (result: seq<bool>)
    ensures |result| == |x|
    // Core NaN detection property: result[i] is true iff x[i] is NaN
    ensures forall i :: 0 <= i < |x| ==> (result[i] <==> IsNaN(x[i]))
    // IEEE 754 fundamental NaN property: NaN ≠ NaN (self-inequality)
    ensures forall i :: 0 <= i < |x| ==> (result[i] ==> x[i] != x[i])
    // Non-NaN values produce false results
    ensures forall i :: 0 <= i < |x| ==> (!IsNaN(x[i]) ==> !result[i])
    // Finite values always produce false results
    ensures forall i :: 0 <= i < |x| ==> (IsFinite(x[i]) ==> !result[i])
    // Complement property: isnan is complement of (finite or infinite)
    ensures forall i :: 0 <= i < |x| ==> (result[i] <==> !(IsFinite(x[i]) || IsInfinite(x[i])))

method Witness()
  ensures false
{
  var r := IsNan([NaN]);
  assert r[0];
}
