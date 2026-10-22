predicate validInput(s: string)
{
    |s| > 0 && (forall i :: 0 <= i < |s| ==> s[i] == ' ' || s[i] == '\n' || ('0' <= s[i] <= '9') || s[i] == '-')
}

predicate validNumber(s: string)
{
    |s| == 0 || (forall i :: 0 <= i < |s| ==> '0' <= s[i] <= '9' || (i == 0 && s[i] == '-'))
}

function countZeros(numbers: seq<int>): int
{
    if |numbers| == 0 then 0
    else (if numbers[0] == 0 then 1 else 0) + countZeros(numbers[1..])
}

function findZeroIndex(numbers: seq<int>): int
    requires |numbers| > 0
    requires countZeros(numbers) == 1
{
    if numbers[0] == 0 then 0
    else if |numbers| > 1 then 1 + findZeroIndex(numbers[1..])
    else 0
}

function parseInts(s: string): seq<int>
    requires |s| > 0
    requires validInput(s)
{
    parseIntsHelper(s, 0, "", [])
}

function generateOutput(numbers: seq<int>): string
{
    generateOutputHelper(numbers, 0, "")
}

// <vc-helpers>
function parseIntsHelper(s: string, i: int, current: string, acc: seq<int>): seq<int>
    requires 0 <= i <= |s|
    requires validInput(s)
    requires validNumber(current)
    decreases |s| - i
{
    if i == |s| then
        if |current| > 0 then acc + [stringToInt(current)]
        else acc
    else if s[i] == ' ' || s[i] == '\n' then
        if |current| > 0 then
            parseIntsHelper(s, i + 1, "", acc + [stringToInt(current)])
        else
            parseIntsHelper(s, i + 1, "", acc)
    else
        // Need to ensure current + [s[i]] is a valid number
        if |current| == 0 && s[i] == '-' then
            parseIntsHelper(s, i + 1, [s[i]], acc)
        else if |current| == 0 && '0' <= s[i] <= '9' then
            parseIntsHelper(s, i + 1, [s[i]], acc)
        else if |current| > 0 && '0' <= s[i] <= '9' then
            parseIntsHelper(s, i + 1, current + [s[i]], acc)
        else
            // This case shouldn't happen with valid input, but we need completeness
            acc
}

function stringToInt(s: string): int
    requires |s| > 0
    requires validNumber(s)
{
    if |s| == 1 then
        if s[0] == '-' then 0
        else (s[0] - '0') as int
    else if s[0] == '-' then
        -stringToIntHelper(s[1..], 0)
    else
        stringToIntHelper(s, 0)
}

function stringToIntHelper(s: string, acc: int): int
    requires |s| > 0
    requires forall i :: 0 <= i < |s| ==> '0' <= s[i] <= '9'
    decreases |s|
{
    if |s| == 1 then
        acc * 10 + (s[0] - '0') as int
    else
        stringToIntHelper(s[1..], acc * 10 + (s[0] - '0') as int)
}

function generateOutputHelper(numbers: seq<int>, i: int, acc: string): string
    requires 0 <= i <= |numbers|
    decreases |numbers| - i
{
    if i == |numbers| then acc
    else 
        var numStr := intToString(numbers[i]);
        if i == |numbers| - 1 then
            generateOutputHelper(numbers, i + 1, acc + numStr)
        else
            generateOutputHelper(numbers, i + 1, acc + numStr + " ")
}

function intToString(n: int): string
{
    if n < 0 then "-" + intToStringHelper(-n, "")
    else if n == 0 then "0"
    else intToStringHelper(n, "")
}

function intToStringHelper(n: nat, acc: string): string
    decreases n
{
    if n == 0 then acc
    else 
        var digit := (n % 10) as char + '0';
        intToStringHelper(n / 10, [digit] + acc)
}
// </vc-helpers>

// <vc-spec>
method solve(input: string) returns (result: string)
    requires validInput(input)
    requires |input| > 0
    ensures var numbers := parseInts(input);
            result == generateOutput(numbers)
// </vc-spec>
// <vc-code>
{
    var numbers := parseInts(input);
    result := generateOutput(numbers);
}
// </vc-code>


lemma Demo() ensures generateOutput([0, 2, 3, 4, 5]) == "0 2 3 4 5" { }
