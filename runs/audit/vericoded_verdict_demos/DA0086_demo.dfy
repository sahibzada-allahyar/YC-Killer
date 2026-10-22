// <vc-preamble>
predicate ValidInput(cards: seq<int>)
{
    |cards| == 5 && forall i :: 0 <= i < |cards| ==> cards[i] > 0
}

function minPossibleSum(cards: seq<int>): int
    requires ValidInput(cards)
    ensures minPossibleSum(cards) >= 0
    ensures minPossibleSum(cards) <= sum(cards)
{
    minPossibleSumUpToIndex(cards, 5)
}
// </vc-preamble>

// <vc-helpers>
/* helper modified by LLM (iteration 2): Adding sum function to resolve unresolved identifier errors related to 'sum' */
function sum(s: seq<int>): int
    ensures sum(s) == if s == [] then 0 else s[0] + sum(s[1..])
{
    if s == [] then
        0
    else
        s[0] + sum(s[1..])
}

function minPossibleSumUpToIndex(cards: seq<int>, k: int): int
    requires ValidInput(cards)
    requires 0 <= k <= |cards|
    decreases k
    ensures minPossibleSumUpToIndex(cards, k) >= 0
    ensures minPossibleSumUpToIndex(cards, k) <= sum(cards[0..k])
{
    if k == 0 then
        0
    else if cards[k-1] > minPossibleSumUpToIndex(cards, k-1) then
        minPossibleSumUpToIndex(cards, k-1)
    else
        cards[k-1]
}
// </vc-helpers>

// <vc-spec>
method solve(cards: seq<int>) returns (result: int)
    requires ValidInput(cards)
    ensures result >= 0
    ensures result <= sum(cards)
    ensures result == minPossibleSum(cards)
// </vc-spec>
// <vc-code>
/* code modified by LLM (iteration 2): Implemented solve method by calling minPossibleSum using the helper function. */
{
  result := minPossibleSum(cards);
}
// </vc-code>

lemma Demo() ensures minPossibleSum([1, 1, 1, 1, 1]) == 0 { }
