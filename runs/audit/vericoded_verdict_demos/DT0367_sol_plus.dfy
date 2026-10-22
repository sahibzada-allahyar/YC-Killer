// <vc-preamble>
/*
 * Computes tangent element-wise for vectors. Equivalent to sin(x)/cos(x) element-wise.
 * The function is undefined when cos(x) = 0 (i.e., x = π/2 + kπ for integer k).
 */

// Uninterpreted trigonometric functions
function sin(x: real): real
{
  0.0  // Dummy implementation for compilation
}

function cos(x: real): real
{
  1.0  // Dummy implementation for compilation
}

function tan(x: real): real
{
  0.0  // Dummy implementation for compilation
}

// Axiom relating tan to sin and cos
// </vc-preamble>

// <vc-helpers>

// </vc-helpers>

// <vc-spec>
lemma {:axiom} tan_definition(x: real)
  requires cos(x) != 0.0
  ensures tan(x) == sin(x) / cos(x)

// Element-wise tangent computation method
method TanElementwise(x: seq<real>) returns (result: seq<real>)
  // Precondition: cosine of each element must be non-zero
  requires forall i :: 0 <= i < |x| ==> cos(x[i]) != 0.0
  // Postcondition: result has same length as input
  ensures |result| == |x|
  // Postcondition: each element is the tangent of corresponding input element
  ensures forall i :: 0 <= i < |x| ==> result[i] == tan(x[i])
  // Postcondition: each element equals sin(x)/cos(x) for corresponding input
  ensures forall i :: 0 <= i < |x| ==> result[i] == sin(x[i]) / cos(x[i])
// </vc-spec>
// <vc-code>
{
  result := [];
  var i := 0;
  while i < |x|
    invariant 0 <= i <= |x|
    invariant |result| == i
    invariant forall j :: 0 <= j < i ==> result[j] == tan(x[j])
    invariant forall j :: 0 <= j < i ==> result[j] == sin(x[j]) / cos(x[j])
  {
    tan_definition(x[i]);
    result := result + [tan(x[i])];
    i := i + 1;
  }
}
// </vc-code>

lemma StubCheck() ensures tan(1.0) == 0.0 && cos(1.0) == 1.0 {}
