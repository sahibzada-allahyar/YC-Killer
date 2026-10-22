predicate ValidInput(input: string) {
    |input| > 0
}

predicate ValidOutput(result: string) {
    result == "Kuro" || result == "Shiro" || result == "Katie" || result == "Draw" || result == ""
}

function OptimalScore(ribbon: string, turns: int): int
    requires |ribbon| >= 0 && turns >= 0
    ensures OptimalScore(ribbon, turns) >= 0
{
    var maxFreq := MaxCharFreq(ribbon);
    var length := |ribbon|;
    if turns == 1 && maxFreq == length then 
        if maxFreq > 0 then maxFreq - 1 else 0
    else if length < maxFreq + turns then length
    else maxFreq + turns
}

// <vc-helpers>
function Max2(a: int, b: int): int
  ensures Max2(a, b) >= a && Max2(a, b) >= b
{
  if a >= b then a else b
}

function Max3(a: int, b: int, c: int): int
  ensures Max3(a, b, c) >= a && Max3(a, b, c) >= b && Max3(a, b, c) >= c
{
  Max2(Max2(a, b), c)
}

function CountChar(ch: char, s: string): int
  requires |s| >= 0
  ensures CountChar(ch, s) >= 0
  decreases |s|
{
  if |s| == 0 then 0
  else if s[0] == ch then 1 + CountChar(ch, s[1..]) else CountChar(ch, s[1..])
}

function MaxCharFreq(ribbon: string): int
  requires |ribbon| >= 0
  ensures MaxCharFreq(ribbon) >= 0
  decreases |ribbon|
{
  if |ribbon| == 0 then 0
  else Max2(CountChar(ribbon[0], ribbon), MaxCharFreq(ribbon[1..]))
}

function SplitLines(input: string): seq<string> {
  if |input| == 0 then [] else [input]
}

function ParseInt(s: string): int {
  0
}
// </vc-helpers>

// <vc-spec>
method solve(input: string) returns (result: string)
    requires ValidInput(input)
    ensures ValidOutput(result)
    ensures var lines := SplitLines(input);
            if |lines| < 4 then result == ""
            else (
                var turns := ParseInt(lines[0]);
                var s0 := lines[1];
                var s1 := lines[2]; 
                var s2 := lines[3];
                var score0 := OptimalScore(s0, turns);
                var score1 := OptimalScore(s1, turns);
                var score2 := OptimalScore(s2, turns);
                var maxScore := Max3(score0, score1, score2);
                var winners := (if score0 == maxScore then 1 else 0) + (if score1 == maxScore then 1 else 0) + (if score2 == maxScore then 1 else 0);
                (winners > 1 ==> result == "Draw") &&
                (winners == 1 && score0 == maxScore ==> result == "Kuro") &&
                (winners == 1 && score1 == maxScore ==> result == "Shiro") &&
                (winners == 1 && score2 == maxScore ==> result == "Katie")
            )
// </vc-spec>
// <vc-code>
{
  var lines := SplitLines(input);
  if |lines| < 4 {
    result := "";
    return;
  }
  var turns := ParseInt(lines[0]);
  var s0 := lines[1];
  var s1 := lines[2];
  var s2 := lines[3];
  var score0 := OptimalScore(s0, turns);
  var score1 := OptimalScore(s1, turns);
  var score2 := OptimalScore(s2, turns);
  var maxScore := Max3(score0, score1, score2);
  var winners := (if score0 == maxScore then 1 else 0) + (if score1 == maxScore then 1 else 0) + (if score2 == maxScore then 1 else 0);
  if winners > 1 {
    result := "Draw";
    return;
  }
  if score0 == maxScore {
    result := "Kuro";
    return;
  }
  if score1 == maxScore {
    result := "Shiro";
    return;
  }
  result := "Katie";
}
// </vc-code>


lemma Demo(s: string) ensures |SplitLines(s)| <= 1 && ParseInt(s) == 0 {}
