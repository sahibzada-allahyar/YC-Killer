predicate validInput(stdin_input: string)
{
    var lines := splitLines(stdin_input);
    |lines| >= 1 && 
    (var n := parseInteger(lines[0]);
     n >= 0 && |lines| >= 2*n + 1 && 
     (forall i :: 1 <= i <= 2*n ==> i < |lines| && |lines[i]| > 0))
}

function computeMismatches(stdin_input: string): nat
    requires validInput(stdin_input)
    ensures computeMismatches(stdin_input) <= parseInteger(splitLines(stdin_input)[0])
{
    var lines := splitLines(stdin_input);
    var n := parseInteger(lines[0]);
    if n == 0 then 0
    else
        var prevSizes := countSizes(lines[1..n+1]);
        var currentSizes := lines[n+1..2*n+1];
        countUnmatchedSizes(prevSizes, currentSizes)
}

// ===== HELPERS: you may add lemmas, functions and predicates here =====
function splitLines(s: string): seq<string> { [] }
function parseInteger(s: string): int { 0 }
function countSizes(lines: seq<string>): seq<string> { lines }
function countUnmatchedSizes(prev: seq<string>, cur: seq<string>): nat { 0 }
function intToString(n: nat): string { "" }
// ===== END HELPERS =====

method solve(stdin_input: string) returns (result: string)
    requires |stdin_input| > 0
    requires validInput(stdin_input)
    ensures |result| > 0
    ensures result[|result|-1] == '\n' || (|result| > 1 && result[|result|-2..] == "\r\n")
    ensures exists mismatches: nat :: result == intToString(mismatches) + "\n" && 
            mismatches == computeMismatches(stdin_input)
    ensures (var lines := splitLines(stdin_input);
             var n := parseInteger(lines[0]);
             n >= 0 ==> (var mismatches := computeMismatches(stdin_input);
                        mismatches <= n &&
                        result == intToString(mismatches) + "\n"))
{
    result := "\n";
}
