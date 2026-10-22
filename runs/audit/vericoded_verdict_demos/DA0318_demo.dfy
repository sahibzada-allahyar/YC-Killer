predicate ValidInput(n: int, l: int, r: int)
{
    n >= 1 && l >= 1 && r >= l && r <= n && r <= 20
}

function MinSumCalculation(n: int, l: int): int
    requires n >= 1 && l >= 1
{
    var start_power := Power(2, l - 1);
    SumWithDecreasingPowers(n, start_power)
}

function MaxSumCalculation(n: int, r: int): int
    requires n >= 1 && r >= 1
{
    var max_power := Power(2, r - 1);
    SumWithIncreasingPowers(n, max_power)
}

// <vc-helpers>
function Power(base: int, exp: nat): int
    requires base >= 1
    ensures Power(base, exp) >= 1
{
    if exp == 0 then 1
    else base * Power(base, exp - 1)
}

function SumWithDecreasingPowers(n: int, start_power: int): int
    requires n >= 1
    requires start_power >= 1
    ensures SumWithDecreasingPowers(n, start_power) >= n
{
    if n == 1 then
        start_power
    else if start_power == 1 then
        start_power + SumWithDecreasingPowers(n - 1, 1)
    else
        start_power + SumWithDecreasingPowers(n - 1, start_power / 2)
}

function SumWithIncreasingPowers(n: int, max_power: int): int
    requires n >= 1
    requires max_power >= 1
    ensures SumWithIncreasingPowers(n, max_power) >= n
{
    if n == 1 then
        max_power
    else
        max_power + SumWithIncreasingPowers(n - 1, max_power * 2)
}

lemma PowerPositive(base: int, exp: nat)
    requires base >= 1
    ensures Power(base, exp) >= 1
{
    // Proof by induction on exp
}

lemma PowerMonotonic(base: int, exp1: nat, exp2: nat)
    requires base >= 2
    requires exp1 <= exp2
    ensures Power(base, exp1) <= Power(base, exp2)
{
    if exp1 == exp2 {
        // Base case: equal exponents
    } else if exp1 == 0 {
        // Power(base, 0) = 1 <= Power(base, exp2)
        PowerPositive(base, exp2);
    } else {
        // exp1 > 0 && exp1 < exp2
        PowerMonotonic(base, exp1 - 1, exp2 - 1);
        assert Power(base, exp1 - 1) <= Power(base, exp2 - 1);
        assert Power(base, exp1) == base * Power(base, exp1 - 1);
        assert Power(base, exp2) == base * Power(base, exp2 - 1);
        assert base >= 2;
        assert base * Power(base, exp1 - 1) <= base * Power(base, exp2 - 1);
    }
}

lemma SumComparisonHelper(n: int, p1: int, p2: int)
    requires n >= 1
    requires p1 >= 1
    requires p2 >= 1
    requires p1 <= p2
    ensures SumWithDecreasingPowers(n, p1) <= SumWithIncreasingPowers(n, p2)
    decreases n
{
    if n == 1 {
        // Base case: p1 <= p2
    } else {
        if p1 == 1 {
            SumComparisonHelper(n - 1, 1, p2 * 2);
        } else {
            SumComparisonHelper(n - 1, p1 / 2, p2 * 2);
        }
    }
}

lemma MinMaxOrdering(n: int, l: int, r: int)
    requires ValidInput(n, l, r)
    ensures MinSumCalculation(n, l) <= MaxSumCalculation(n, r)
{
    var start_power := Power(2, l - 1);
    var max_power := Power(2, r - 1);
    
    assert l <= r;
    assert start_power == Power(2, l - 1);
    assert max_power == Power(2, r - 1);
    
    if l <= r {
        if l == 1 {
            assert l - 1 == 0;
            assert r - 1 >= 0;
            PowerMonotonic(2, 0, r - 1);
        } else {
            assert l - 1 >= 0;
            assert r - 1 >= l - 1;
            PowerMonotonic(2, l - 1, r - 1);
        }
        assert Power(2, l - 1) <= Power(2, r - 1);
    }
    
    SumComparisonHelper(n, start_power, max_power);
}
// </vc-helpers>

// <vc-spec>
method solve(n: int, l: int, r: int) returns (min_sum: int, max_sum: int)
    requires ValidInput(n, l, r)
    ensures min_sum > 0
    ensures max_sum > 0
    ensures min_sum <= max_sum
    ensures min_sum == MinSumCalculation(n, l)
    ensures max_sum == MaxSumCalculation(n, r)
// </vc-spec>
// <vc-code>
{
    var start_power := Power(2, l - 1);
    min_sum := SumWithDecreasingPowers(n, start_power);
    
    var max_power := Power(2, r - 1);
    max_sum := SumWithIncreasingPowers(n, max_power);
    
    MinMaxOrdering(n, l, r);
}
// </vc-code>


lemma Demo() ensures MaxSumCalculation(3, 2) == 14 { }
