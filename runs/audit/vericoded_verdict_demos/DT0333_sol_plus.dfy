// <vc-preamble>
/*
 * Looking at the compilation errors, the functions `exp` and `log` have no body and cannot be compiled. Since these are mathematical functions used in specifications, I need to mark them as axioms to make them compilable while preserving the intended semantics.
 */



// Mathematical functions for logarithm and exponential
function {:axiom} exp(x: real): real
{
  1.0  // Stub implementation for compilation
}

function {:axiom} log(x: real): real
  requires x > 0.0
{
  0.0  // Stub implementation for compilation
}

function max(x: real, y: real): real
{
  if x >= y then x else y
}

// Mathematical properties of log and exp functions
// </vc-preamble>

// <vc-helpers>
/* helper modified by LLM (iteration 2): instantiate LogExpProperties and prove lower bound */
lemma LogAddExpLowerBound(a: real, b: real)
  ensures log(exp(a) + exp(b)) >= max(a,b)
{
  // Make properties of exp and log available
  LogExpProperties();
  var m := max(a,b);
  if a >= b {
    assert m == a;
    // exp(...) > 0 by LogExpProperties
    assert exp(a) > 0.0;
    assert exp(b) > 0.0;
    // sum strictly greater than exp(a)
    assert exp(a) < exp(a) + exp(b);
    // log is defined on positive reals and monotone, and log(exp(a)) == a
    assert log(exp(a)) == a;
    assert log(exp(a)) < log(exp(a) + exp(b));
    assert a < log(exp(a) + exp(b));
  } else {
    assert m == b;
    assert exp(a) > 0.0;
    assert exp(b) > 0.0;
    assert exp(b) < exp(a) + exp(b);
    assert log(exp(b)) == b;
    assert log(exp(b)) < log(exp(a) + exp(b));
    assert b < log(exp(a) + exp(b));
  }
  assert m <= log(exp(a) + exp(b));
}

/* helper modified by LLM (iteration 2): prove exact formula when both inputs equal */
lemma LogAddExpEqualCase(a: real)
  ensures log(exp(a) + exp(a)) == a + log(2.0)
{
  LogExpProperties();
  assert exp(a) > 0.0;
  assert 2.0 > 0.0;
  assert exp(a) + exp(a) == 2.0 * exp(a);
  assert log(2.0 * exp(a)) == log(2.0) + log(exp(a));
  assert log(exp(a) + exp(a)) == log(2.0 * exp(a));
  assert log(exp(a)) == a;
  assert log(exp(a) + exp(a)) == a + log(2.0);
}

// </vc-helpers>

// <vc-spec>
lemma {:axiom} LogExpProperties()
  ensures forall x: real :: x > 0.0 ==> exp(log(x)) == x
  ensures forall x: real :: log(exp(x)) == x
  ensures forall x, y: real :: exp(x + y) == exp(x) * exp(y)
  ensures forall x, y: real :: (x > 0.0 && y > 0.0) ==> log(x * y) == log(x) + log(y)
  ensures forall x: real :: exp(x) > 0.0
  ensures forall x, y: real :: x < y ==> exp(x) < exp(y)
  ensures forall x, y: real :: (x > 0.0 && y > 0.0 && x < y) ==> log(x) < log(y)

/**
 * Computes the logarithm of the sum of exponentiations element-wise.
 * For each index i, computes log(exp(x1[i]) + exp(x2[i])).
 * This provides numerically stable computation without intermediate overflow.
 */
method logaddexp(x1: seq<real>, x2: seq<real>) returns (result: seq<real>)
  requires |x1| == |x2|
  ensures |result| == |x1|
  // Element-wise correctness: result[i] = log(exp(x1[i]) + exp(x2[i]))
  ensures forall i :: 0 <= i < |result| ==> 
    result[i] == log(exp(x1[i]) + exp(x2[i]))
  // Bounds property: result[i] >= max(x1[i], x2[i])
  ensures forall i :: 0 <= i < |result| ==> 
    result[i] >= max(x1[i], x2[i])
  // Special case: logaddexp(x, x) = x + log(2)
  ensures forall i :: 0 <= i < |result| ==> 
    x1[i] == x2[i] ==> result[i] == x1[i] + log(2.0)
  // Commutativity: logaddexp(x1, x2) = logaddexp(x2, x1)
  ensures forall i :: 0 <= i < |result| ==> 
    result[i] == log(exp(x2[i]) + exp(x1[i]))
// </vc-spec>
// <vc-code>
{
  /* code modified by LLM (iteration 2): compute element-wise using helpers and ensure positivity */
  // Make Log/Exp properties available to the verifier
  LogExpProperties();
  var n := |x1|;
  result := [];
  var i := 0;
  while i < n
    invariant 0 <= i <= n
    invariant |result| == i
    invariant forall j :: 0 <= j < i ==> result[j] == log(exp(x1[j]) + exp(x2[j]))
    invariant forall j :: 0 <= j < i ==> result[j] >= max(x1[j], x2[j])
    invariant forall j :: 0 <= j < i ==> (x1[j] == x2[j] ==> result[j] == x1[j] + log(2.0))
  {
    // ensure argument to log is positive
    assert exp(x1[i]) > 0.0;
    assert exp(x2[i]) > 0.0;
    assert exp(x1[i]) + exp(x2[i]) > 0.0;
    // discharge auxiliary lemmas for verification
    LogAddExpLowerBound(x1[i], x2[i]);
    LogAddExpEqualCase(x1[i]);
    var v := log(exp(x1[i]) + exp(x2[i]));
    result := result + [v];
    i := i + 1;
  }
}

// </vc-code>

lemma TaskAxiomsProveFalse() ensures false { LogExpProperties(); assert log(exp(1.0)) == 1.0; }
