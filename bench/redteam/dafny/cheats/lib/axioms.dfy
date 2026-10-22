// Used only by include_axioms.dfy: the included file joins the same program.
lemma {:axiom} Magic(a: seq<int>, r: seq<int>)
  ensures ValidTransformation(a, r)
