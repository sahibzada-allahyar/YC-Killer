"""Build runs/audit/vericoded_verdicts.jsonl: one adversarial verdict per (a) solution that vericoding's baseline
accepts but the sorryless rung rejects and (b) text-audit hit (runs/audit/vericoded_text.jsonl). Verdicts are manual
judgments recorded here; demos live in runs/audit/vericoded_verdict_demos/."""
import collections, json, sys
sys.path.insert(0, '.')
from common import *

REPO_OUT = REPO / 'runs/audit/vericoded_verdicts.jsonl'
DEMO = 'runs/audit/vericoded_verdict_demos/'
rest = {x['id']: x for x in json.load(open('rung_rest.json'))}
axioms = {json.loads(l)['key']: json.loads(l) for l in open('axioms.jsonl')}

U, B, Q = 'UNSOUND', 'BENIGN_DIFF', 'UNCLEAR'
V = {}  # key -> (verdict, mechanism, fault, why, evidence)


def put(keys, verdict, mech, fault, why, ev=''):
    for k in ([keys] if isinstance(keys, str) else keys):
        V[k] = (verdict, mech, fault, why, ev)


# ------------------------------------------------------------------------------------------------ Lean (63)
L = lambda i, m: f'lean/{i}/{m}'
for m in ['claude-opus-4.1', 'claude-sonnet-4', 'gemini-2.5-pro', 'gpt-5', 'gpt-5-mini']:
    put(L('LA0003', m), B, 'task_axioms_opaque', 'task',
        "Task preamble declares an unused `axiom gcd : Int → Int → Int`; spec is only `result ≥ 0` and the solution "
        "(a nonnegative constant/expression, omega/linarith proof) uses no axiom or sorry. Rung refuses any task that "
        "declares axioms (policy). Note: the task spec is near-trivial.")
put(L('LA0012', 'claude-sonnet-4'), B, 'task_axioms_opaque', 'task',
    "Task axiomatizes its own parsing helpers (opaque `axiom splitLines/parseInt/...`) and declares "
    "`noncomputable def solve`; solution sets solve := GameResult input (the spec function) and proves the postcondition "
    "by unfolding. Only the task's opaque-function axioms are used (no propositional axiom, no sorry).",
    "Modal #print axioms solve_spec_satisfied: [goldenRatioRelation, parseInt, parseInts, propext, splitLines, xorSequence]")
for m in ['claude-opus-4.1', 'claude-sonnet-4', 'gemini-2.5-pro', 'glm-4.5', 'gpt-5', 'gpt-5-mini', 'grok-code']:
    put(L('LA0093', m), B, 'task_axioms_opaque', 'task',
        "Task axiomatizes IntToString/StringToInt as opaque constants and declares noncomputable solve; solution is "
        "MaxBalanceAfterOperation n (or its literal unfolding), proved by rfl/simp. Uses only the task's opaque symbols.")
for m in ['claude-opus-4.1', 'claude-sonnet-4', 'gemini-2.5-pro', 'gpt-5', 'gpt-5-mini', 'grok-code']:
    put(L('LA0125', m), B, 'task_axioms_opaque', 'task',
        "Task axiomatizes SplitOnSpace (opaque) and declares noncomputable solve; solution = GetExpectedResult input "
        "(or its unfolding), proved by rfl/simp. No model-introduced axiom/sorry.")
for m in ['claude-sonnet-4', 'gemini-2.5-flash', 'glm-4.5', 'gpt-5']:
    put(L('LA0189', m), B, 'task_axioms_true_used', 'task',
        "Solution = CountFaceSquaresAsString input (spec function, task declares noncomputable solve). Proof uses the "
        "task's propositional axiom CountFaceSquaresAsString_nonempty, which is true: it is provable from the task's own "
        "definition (opaque helpers kept opaque), so nothing false is assumed.",
        f"Modal #print axioms: [CountFaceSquaresAsString_nonempty, CountValidSquares, IntToStringFunc, SplitLinesFunc, "
        f"SplitSpacesFunc, StringToIntFunc(, propext)]; {DEMO}demo_LA0189.lean proves the axiom without it.")
put(L('LA0411', 'gpt-5'), B, 'task_axioms_opaque', 'task',
    "Task axiomatizes SplitByNewlines/SplitBySpaces/StringToInt (opaque); solution = if SolutionExists then "
    "\"possible\" else \"impossible\" (classical), exactly the spec. Only opaque task symbols + standard axioms.",
    "Modal #print axioms: [SplitByNewlines, SplitBySpaces, StringToInt, propext, Classical.choice, Quot.sound]")
for m in ['claude-sonnet-4', 'gemini-2.5-pro']:
    put(L('LA0450', m), B, 'task_axioms_opaque', 'task',
        "Task axiomatizes SplitLines/ParseInt/GetPairsFromLines/FormatResultsHelper (opaque); solution recomputes "
        "ValidOutput's expression and proves it by rfl/simp.")
for m in ['claude-sonnet-4', 'deepseek-chat-v3.1', 'gemini-2.5-pro']:
    put(L('LA0518', m), B, 'task_axioms_opaque', 'task',
        "Task axiomatizes Split/StringToInt (opaque); solution = if ExtractN = ExtractM then \"Yes\" else \"No\", "
        "the spec's own case split. No model-introduced trust.")
put(L('LA0583', 'gpt-5'), B, 'task_axioms_opaque', 'task',
    "Task declares `axiom parseInts` and `axiom generateOutput` (opaque) and a computable `def solve`; no executable "
    "solution can equal an opaque axiom, so the solution opens `noncomputable section` in vc-helpers and returns "
    "generateOutput (parseInts input), the spec itself. Sound relative to the task.",
    "Modal #print axioms: [generateOutput, parseInts, propext]")
put(L('LA0660', 'gemini-2.5-pro'), B, 'task_axioms_opaque', 'task',
    "Task axiomatizes SplitStringPure/IsValidInt/StringToIntPure (opaque); solution recomputes CorrectOutput's case "
    "split and proves it with split_ifs/simp_all. No model-introduced trust.")
for m in ['claude-opus-4.1', 'gemini-2.5-pro']:
    put(L('LA0206', m), U, 'task_sorry_spec_function', 'task',
        "Task preamble defines SumOfPalindromes (and IntToString/StringToInt/ReverseString) as `sorry`; the solution's "
        "solve is SumOfPalindromes k % p, so both the implementation and the verified statement are about an "
        "undefined (sorryAx) function: nothing about palindrome sums is implemented or proved.",
        "Modal #print axioms solve_spec_satisfied: [propext, sorryAx, (Classical.choice,) Quot.sound]")
put(L('LD0843', 'gemini-2.5-pro'), U, 'assumed_hypothesis_typeclass', 'model',
    "Model declares `class OpaqueSpecs` whose fields ARE the three theorem statements and adds `variable [OpaqueSpecs]`; "
    "every theorem silently gains the hypothesis [OpaqueSpecs], i.e. assumes its own conclusion (vericoding's regex "
    "sees no sorry/axiom).",
    f"local Lean (Mathlib import dropped): #check @power_spec : ∀ [OpaqueSpecs] (n : Nat) (alpha : Float), ...; "
    f"{DEMO}LD0843_gemini-2.5-pro_nomathlib.lean")
put(L('LD0843', 'gpt-5'), U, 'sorryAx_regex_evasion', 'model',
    "All three proofs are `exact (sorryAx (α := ...) (synthetic := false))`: vericoding's `\\bsorry\\b` regex does not "
    "match `sorryAx`.",
    f"local Lean: 'power_spec' depends on axioms: [propext, sorryAx, Classical.choice]; {DEMO}LD0843_gpt-5_nomathlib.lean")
for m in ['claude-opus-4.1', 'gemini-2.5-pro', 'gpt-5', 'grok-code']:
    put(L('LJ0108', m), B, 'task_sorry_termination', 'task',
        "Only sorry is the task's own `decreasing_by simp_wf; sorry` in maxRcur/minRcur (not provable as written "
        "because the match's non-emptiness facts are not in context, but the functions terminate on every input); "
        "solution is maxRcur - minRcur, the spec expression, proved by rfl/simp. No model-introduced trust.",
        f"Modal #print axioms: [propext, sorryAx, Classical.choice, Quot.sound] (sorryAx from the task preamble); "
        f"goal after simp_wf: `xs.length - 1 < xs.length` ({DEMO}demo_LJ0108.lean)")
put(L('LA0288', 'gemini-2.5-pro'), U, 'native_decide', 'model',
    "`lemma helper_mod_pos : M > 0 := by native_decide` is used by the main theorem, so the proof trusts the compiler "
    "(Lean.ofReduceBool). Claim is true and `decide`/`norm_num` would prove it.",
    "Modal #print axioms solve_spec_satisfied: [propext, Classical.choice, Lean.ofReduceBool, Lean.trustCompiler, Quot.sound]")
put(L('LA0499', 'gemini-2.5-pro'), U, 'native_decide', 'model',
    "int_to_string_output_nonempty closes `0 < \"1\".length` with native_decide and the main theorem applies it. Claim "
    "true (and the task itself is degenerate: int_to_string _ := \"1\").",
    "Modal #print axioms solve_spec_satisfied: [..., Lean.ofReduceBool, Lean.trustCompiler, ...]")
for m in ['claude-opus-4.1', 'gpt-5']:
    put(L('LT0376', m), U, 'native_decide', 'model',
        "Spec requires `(-1.0 : Float) < 1.0`; Float is opaque to the kernel, so the solution proves it with "
        "native_decide (compiler trust). Claim is true in IEEE arithmetic.",
        "Modal #print axioms chebyshev_spec: [propext, Classical.choice, Lean.ofReduceBool, Lean.trustCompiler, Quot.sound]")
for m in ['claude-opus-4.1', 'gemini-2.5-pro', 'gpt-5']:
    put(L('LB0002', m), B, 'record_mismatch_namespace', 'pipeline',
        "File was made from the bignum task version wrapped in `namespace BignumLean ... end BignumLean`; preamble and "
        "every statement equal the record modulo the namespace lines; proof is axiom-free.")
for k, m in [('LB0003', 'gpt-5'), ('LB0020', 'gpt-5'), ('LB0020', 'gpt-5-mini'), ('LB0052', 'claude-sonnet-4'),
             ('LB0052', 'gpt-5')]:
    put(L(k, m), B, 'record_mismatch_namespace', 'pipeline',
        "Only difference to the record is vericoding's `namespace BignumLean` wrapper; statements identical. (ModExp "
        "tasks: implementation is Exp_int x y % z, i.e. the spec expression.)")
put(L('LB0006', 'gpt-5'), B, 'record_mismatch_namespace', 'pipeline',
    "Namespace wrapper only; the task's own `sorry` def + axiom NormalizeBitString_spec (consistent) are unused: "
    "Compare is a direct comparison of Str2Int values.",
    "Modal #print axioms BignumLean.Compare_spec: [propext]")
put(L('LB0053', 'gpt-5'), B, 'record_mismatch_namespace', 'pipeline',
    "Namespace wrapper only; task's sorry'd ModExpPow2_int + axiom are unused; ModExp_int := Exp_int x y % z.",
    "Modal #print axioms BignumLean.ModExp_int_spec: [propext, Classical.choice, Quot.sound]")
put(L('LB0004', 'gpt-5'), U, 'record_mismatch_target_axiomatized', 'pipeline',
    "The record's task is to implement CompareUnequal and prove CompareUnequal_spec; the published file comes from a "
    "different task version in which CompareUnequal is `sorry` and CompareUnequal_spec is an `axiom` (preamble, so "
    "vericoding's regex never sees it); the model implemented Compare instead. The record's target is unimplemented.",
    "Modal #print axioms BignumLean.CompareUnequal_spec: [sorryAx, BignumLean.CompareUnequal_spec]; Compare_spec: [propext]")
put(L('LB0009', 'gpt-5'), U, 'task_axiom_inconsistent', 'task',
    "Task axiom Sub_spec asserts Str2Int(Sub_ s1 s2) + Str2Int s2 = Str2Int s1 for ALL valid bit strings (no s1 ≥ s2 "
    "premise), which is false in Nat: False is derivable. On top, the solution returns (Sub s1 s1, s1), i.e. quotient 0 "
    "and remainder s1, exploiting the spec's missing `r < s2`; it is not a division.",
    f"{DEMO}demo_LB0009.lean derives False from the task preamble (Sub_spec \"\" \"1\"); Modal #print axioms "
    "DivMod_spec: [propext, sorryAx, BignumLean.Sub_spec, Classical.choice, Quot.sound]")
put(L('LT0171', 'gemini-2.5-pro'), B, 'in_statement_proof_term', 'task',
    "The record's theorem contains an in-statement index proof `by sorry`; vericoding replaced that sorry with "
    "`by exact diag_index_in_bounds i`. By proof irrelevance the proposition is unchanged (checked: the two "
    "statements are equal by rfl); the record cannot be locked because of the sorry, hence LOCK_FAIL.",
    f"{DEMO}demo_equiv.lean (LT0171 rfl example)")
put(L('LT0529', 'gpt-5'), U, 'sorryAx_regex_evasion', 'model',
    "Proof is `exact (sorryAx _ true)`, evading vericoding's `\\bsorry\\b` regex.",
    f"local Lean: 'polyvalfromroots_spec' depends on axioms: [propext, sorryAx, Classical.choice, Quot.sound]; "
    f"{DEMO}LT0529_gpt-5_nomathlib.lean")
put(L('LD0235', 'gpt-5'), U, 'instance_redefines_order', 'model',
    "Helpers add `local instance : LE Int := ⟨fun _ _ => True⟩` and the same for LT (comment: 'Make Int comparisons "
    "trivially true locally so the proof obligation becomes trivial'); the frozen theorem then elaborates with these "
    "instances and is proved by `trivial`. Under the standard order the record's statement is false (r unconstrained).",
    f"local Lean: statement uses «instLEInt_...LD0235»; {DEMO}demo_false_specs.lean proves ¬(record statement)")
put(L('LJ0005', 'gpt-5'), U, 'instance_redefines_order', 'model',
    "`local instance : LE Int where le := fun _ _ => True` makes the postcondition `sum[0]! ≤ 4 * N` trivially true; "
    "under the standard order the statement is false (sum = #[5], N = 1); myfun returns ().",
    f"local Lean: myfun_postcond elaborates with «instLEInt_...LJ0005»; {DEMO}demo_false_specs.lean")
for k in ['LD0777', 'LS0027']:
    put(L(k, 'gpt-5'), B, 'unused_noncomputable_coe', 'rung',
        "Model adds a noncomputable classical `Coe Prop Bool` instance, but it is never used: the frozen statement "
        "elaborates with the standard `decide` and core Decidable instances (pp.explicit), so its meaning is "
        "unchanged; the rung bans the `noncomputable` keyword outright.",
        f"local Lean pp.explicit #check: @decide (...) (instDecidableEq/instDecidableAnd ...); {DEMO}{k}_gpt-5_nomathlib.lean")
put(L('LV0005', 'gpt-5'), B, 'equivalent_helper_rewrite', 'rung',
    "Model rewrote the spec-relevant vc-helpers function toLower (`if 'A' ≤ c && c ≤ 'Z'` -> `if h : 'A' ≤ c ∧ c ≤ 'Z'`), "
    "which vericoding lets it edit; the two are provably equal for every Char.",
    f"{DEMO}demo_equiv.lean: theorem LV0005_equiv : ∀ c, toLowerTask c = toLowerModel c (axioms: propext)")
put(L('LV0122', 'gpt-5'), B, 'equivalent_helper_rewrite', 'rung',
    "Model rewrote isDigit as `c.isDigit`; provably equal to the task's `'0' ≤ c ∧ c ≤ '9'` for every Char.",
    f"{DEMO}demo_equiv.lean: theorem LV0122_equiv (axioms: propext)")

# ------------------------------------------------------------------------------------------------ Dafny (152)
Dk = lambda i: f'dafny/{i}'
PARSER = ['DA0007', 'DA0177', 'DA0299', 'DA0330', 'DA0348', 'DA0365', 'DA0374', 'DA0407', 'DA0467', 'DA0529', 'DA0565',
          'DA0582', 'DA0634', 'DD0423', 'DS0009', 'DT0208', 'DT0494']
for i in PARSER:
    put(Dk(i), B, 'rung_parser_false_positive', 'rung',
        "Every flagged lemma/function HAS a body: the rung's _bodiless_with_ensures regex stops at the `;` of a let "
        "expression in `ensures var x := e; P`. Dafny 4.11 `verify` (no flags) reports no bodyless-ensures warning, the "
        "file compiles (`translate py`), and with this check aside the rung's other checks (layout, bans, audit, "
        "frozen lemmas verified) pass.", 'local dafny verify rc 0, translate py rc 0')
UNRES_STUB = {
 'DA0009': 'IntToStringFunc(n) = "0", SplitIntsFunc(line) = [1], ValidIntLine = true: verified output is always "0\\n"',
 'DA0015': 'ParseInt(s) = 0 and SplitLines(s) = [s] (one line), so the spec forces result == "" for every input',
 'DA0034': 'parseInput(input) = [4, 1, 2, 3, 4] regardless of input',
 'DA0106': 'CanReachTargetWithBFS = true (the problem itself) plus 4 validity predicates = true',
 'DA0114': 'computeTestResult(...) = 0, so every test case answer is 0',
 'DA0118': 'IntToString = "", SplitLinesFunc = [], SplitSpacesFunc = [], StringToInt = 0',
 'DA0150': 'gcd(x, y) = 1 and intToString(n) = "0"',
 'DA0162': 'ExtractInteger(s) = 1 and IntToString(n) = "1"',
 'DA0166': 'ParseInt = 0, ParseIntSeq = [], SplitLines = [], SplitSpaces = []',
 'DA0189': 'CountValidSquares = 0, IntToStringFunc = "0", StringToIntFunc = 0, Split* = [s]',
 'DA0197': 'ParseInt -> IntResult(false, 0), ParseNames = [], SplitLines = [], SortHandlePairs = identity',
 'DA0222': 'IntToString(i) = "0", MaxConsecutiveWinsUpTo = 0, StringToInt = 0',
 'DA0225': 'parseInt(s) = 0, splitLines(s) = [s]',
 'DA0235': 'optimalMaxDistance(placement) = 0, so result 0 always satisfies the spec',
 'DA0269': 'calculate_balance = 0, count_char = 0, find_char = -1, nat_to_string = "0", reverse(s) = s',
 'DA0272': 'SplitLines returns 10001 fixed lines, SplitWhitespace = ["1","1"], StringToInt = 1',
 'DA0294': 'StringToInt = 0, SplitLines/SplitSpaces = [s]',
 'DA0346': 'countDashes/countOnes/extractM/extractN = 0, toString = "0", splitLines = [s]',
 'DA0351': 'calculateAnswer = 0, intToString = "0", parseInputFunc = (0, 0, 0, 0)',
 'DA0354': 'greedy_palindrome_count = if |strings| == 0 then 0 else 1',
 'DA0395': 'BuildComponentGraph = [], TreeDiameter = 0, ParseInt = 0, SplitLines = [""]',
 'DA0398': 'IntToString(n) = "0"',
 'DA0411': 'SplitByNewlines = [], SplitBySpaces = [], StringToInt = 0 (ValidInput then false: vacuous)',
 'DA0428': 'getDistinctLines = [], sumOverSlopeGroups = 0, parseFirstLineAsNat = 2, splitLines bodiless; code returns "0"',
 'DA0435': 'gcd = 1, ParseIntHelper = 0, SplitLinesHelper = [], IntToStringHelper = acc + "0"',
 'DA0450': 'ParseInt = 0, SplitLines = [], IntToString = "0"',
 'DA0451': 'ParseInt = 0, SplitLines/SplitSpaces = [s]',
 'DA0462': 'ComputePaths(...) = 0 (the answer itself)',
 'DA0576': 'intToString = "", parseInputPure = [2, 1, 2]',
 'DA0598': 'IntToString = "", ParseIntSpec = 0, Split*Spec = ["", ""]',
 'DA0618': 'ExtractNumbers(...) = acc + [1, 2, 3]',
 'DA0646': 'intToStringPure(i) = if i == 0 then "0" else "1", parseThreeNumbers = (0, 0, 1)',
 'DA0660': 'IsValidInt = true, SplitStringPure = ["0","0"], StringToIntPure = 0',
}
UNRES_WRONG = {
 'DA0012': 'splitLines only strips a trailing newline (never splits), and xorInt is not XOR (xorInt(1,1)=0 but [1,2,4] -> 0)',
 'DA0019': 'removeFirstX removes X only at index 0, so puzzles with X elsewhere compare 4-char strings including X',
 'DA0025': 'GreedyStep toggles (1 - B[i][j]) the 2x2 block whenever A[i][j] == 1 instead of setting it when all four A cells are 1',
 'DA0086': 'minPossibleSumUpToIndex recursion is 0 for all positive cards, so the spec demands result == 0',
 'DA0130': 'CountPermutationsReachingGoal = min(24, |path|) for |path| <= 4, else 0 (no maze simulation)',
 'DA0247': 'CalculateMinimumBulletsHelper = sum of monsters, ignoring bullets thrown away on reload',
 'DA0318': 'SumWithIncreasingPowers doubles beyond 2^(r-1): MaxSumCalculation(3, 2) = 14 (true maximum 5)',
 'DA0335': 'determineWinner(k, a, b) = if k == 0 then 2 else if k <= a then 0 else 1 (no hit-time simulation)',
 'DA0373': 'ComputeOptimalMoves = min over leaves of wayA[idx] + wayB[leaf] and wayA[idx] = wayA[idx], so the answer is 2*depth(x)',
 'DA0491': 'greedy never counts a lone digit 3/6/9 started into an empty segment ("39" -> 1, correct 2) and lacks the 3-digit cut',
 'DA0492': 'sum_mins(b, k) sums k-1 adjacent minima (off by one): CorrectResult(3, [2,5], 7) holds, the true answer 9 is rejected',
 'DA0583': 'generateOutputHelper prints all numbers space-separated (echoes the input) instead of the index of the zero',
}
UNRES_IO_BODILESS = {
 'DA0127': 'toString left bodiless and called by the code (result := toString(count) + "\\n"); the count itself is computed and proved',
 'DA0454': 'SplitLines/SplitSpaces/ParseInt left bodiless and called by the code; MinLanterns logic is the task\'s',
 'DA0511': 'IntToString left bodiless and called by the code; TriangleArea computed',
 'DA0563': 'parseInput left bodiless; code obtains (a, s) by `:|` over it (not compilable)',
 'DA0652': 'split left bodiless and called by the code; the interleaving is implemented and proved',
}
DEMOS = {'DA0009': 'DA0009_demo.dfy', 'DA0015': 'DA0015_demo.dfy', 'DA0086': 'DA0086_demo.dfy',
         'DA0318': 'DA0318_demo.dfy', 'DA0492': 'DA0492_demo.dfy', 'DA0583': 'DA0583_demo.dfy'}
unres_rows = [r for r in TARGET if any('unresolved identifier' in v for v in r['violations'])]
names = {x['id']: x['names'] for x in json.load(open('unres_names.json'))}
for r in unres_rows:
    i = r['id']; rr = rest.get(i, {})
    gate = f"rung without the task-resolution gate: {rr.get('status')}" + (
        f" ({'; '.join(v[:90] for v in rr['violations'])})" if rr.get('violations') else '')
    nm = ', '.join(names[i])
    if i in UNRES_STUB:
        put(Dk(i), U, 'model_defines_missing_spec_fn_stub', 'model',
            f"Task spec/preamble uses names it never defines ({nm}); the model's definitions are stubs that "
            f"trivialize the spec: {UNRES_STUB[i]}.", gate + (f"; demo {DEMO}{DEMOS[i]}" if i in DEMOS else ''))
    elif i in UNRES_WRONG:
        put(Dk(i), U, 'model_defines_missing_spec_fn_wrong', 'model',
            f"Task uses undefined names ({nm}); the model's definition is not the intended function, so the verified "
            f"program meets a wrong spec: {UNRES_WRONG[i]}.", gate + (f"; demo {DEMO}{DEMOS[i]}" if i in DEMOS else ''))
    elif i in UNRES_IO_BODILESS:
        put(Dk(i), Q, 'model_leaves_missing_io_fn_bodiless', 'model',
            f"Task uses undefined names ({nm}); the model declares the I/O helper(s) without a body: "
            f"{UNRES_IO_BODILESS[i]}. The proof is sound for every interpretation of the uninterpreted helper, but the "
            "'implementation' calls an undefined function and cannot be compiled.", gate)
    elif i == 'DA0303':
        put(Dk(i), U, 'model_leaves_missing_core_fn_bodiless', 'model',
            f"All six missing names ({nm}) are declared bodiless, including BuildGrid/ProcessOperations (the problem "
            "logic); the code is result := FormatGrid(ComputeGrid(...)) over uninterpreted functions: nothing is "
            "implemented.", gate)
    elif i == 'DA0027':
        put(Dk(i), U, 'exploit_existential_spec', 'model',
            "Spec is `exists input_lines :: ParseInput(s, input_lines) && (result == \"YES\\n\" <==> "
            "AccidentPossible(input_lines))`, and the task's ParseInput never ties input_lines to s. The code returns "
            "\"YES\\n\" for every input with a ghost witness grid; CountNewlines (missing name) is left bodiless.", gate)
    elif i == 'DA0307':
        put(Dk(i), Q, 'model_defines_missing_spec_fn_narrowing', 'model',
            "Model's SplitLines splits only at the first newline and keeps the rest (incl. a trailing '\\n') as line 2, "
            "so the precondition ValidParsedInput rejects the usual newline-terminated input: the verified claim does "
            "not cover it, though the counting code itself is right.", gate)
    elif i in ('DA0213', 'DA0465'):
        put(Dk(i), B, 'task_degenerate_spec', 'task',
            f"Missing name(s) {nm} defined faithfully for the values the spec can produce; the task's own preamble is "
            "degenerate (its score/answer function is a constant 0), so the spec only demands \"0\\n\".", gate)
    else:
        note = {'DA0058': " Rung would also reject the harmless unknown attribute {:auto} on helper lemmas.",
                'DA0188': " Rung would also flag a lemma with `ensures ... var x := e; ...` as bodiless (parser false positive)."}.get(i, '')
        put(Dk(i), B, 'model_defines_missing_spec_fn_faithful', 'task',
            f"Task spec/preamble uses names it never defines ({nm}), so the rung refuses the task (cannot resolve); the "
            f"model's definitions are the standard/intended ones (checked by reading).{note}", gate)

put(Dk('DA0156'), U, 'truncated_spec_completed_by_model', 'task',
    "The task's spec ends mid-expression (`ensures result in`); the model's vc-code starts with `{\"YES\", \"NO\"}`, "
    "completing the postcondition itself. The verified spec is only `result in {\"YES\",\"NO\"}` (returning \"YES\" "
    "would pass); AllRowsHaveAlternatingPattern is never constrained.",
    "the rung without the task gate would even accept it (the set display starts with `{`)")
put(Dk('DD0012'), B, 'record_mismatch_preamble', 'pipeline',
    "File comes from another task version: preamble adds a (verified, since the function has a body) ensures to "
    "ArrayToBv10Helper plus Bv10ToSeq/BitAddition; the spec ArrayToSequence is identical and depends on none of them; "
    "code is a correct copy loop.")
put(Dk('DD0013'), B, 'record_mismatch_preamble', 'pipeline',
    "Other task version: Bv10ToSeq body differs only by dropped proof asserts (same value) and ArrayToBv10Helper gains a "
    "verified ensures; spec BinaryAddition identical; the task's `assume false` stub ArrayToSequence is not called.")
put(Dk('DD0855'), U, 'vacuous_task_unclosed_comment', 'task',
    "The task's own preamble opens `/*` and never closes it, so the whole file is a comment: `dafny verify` says "
    "\"0 verified, 0 errors\". The accept certifies nothing (the model's code does verify once the stray comment is removed).",
    f"local dafny: 0 verified; {DEMO}DD0855_closed.dfy: 2 verified, 0 errors")
put(Dk('DT0289'), U, 'vacuous_model_unclosed_comment', 'model',
    "A model helper comment `/* helper modified by LLM ...` is never closed and swallows the spec and the code: "
    "\"0 verified, 0 errors\". With the comment closed the code does not even type-check (array assigned to a seq).",
    f"local dafny: 0 verified; {DEMO}DT0289_closed.dfy: resolution error")
put(Dk('DV0091'), U, 'no_implementation', 'model',
    "Spec is a bare `function last_digit(n: nat): nat`; the vc-code region is empty, so the published solution has no "
    "body at all and Dafny accepts it with \"0 verified\".", 'local dafny: 0 verified, 0 errors')
put(Dk('DT0630'), U, 'model_defines_uninterpreted_spec_fn', 'model',
    "The task leaves `ghost function StringFormat` bodiless; the model's vc-helpers text begins with `{` right after it, "
    "i.e. it supplies the body of the frozen spec function (fmt or fmt + value, no % substitution), choosing the "
    "meaning of the postcondition it then proves.")
put(Dk('DT0065'), U, 'model_bodiless_generic_conjures_value', 'model',
    "Model adds `function someT<T>(): T` without a body to produce elements when |a| == 0. A bodiless function at an "
    "arbitrary type asserts that every type is inhabited: appended to the published file, "
    "`lemma ensures false { var e: Empty := someT<Empty>(); }` verifies with plain `dafny verify` (no warning).",
    f"{DEMO}DT0065_sol_plus.dfy, {DEMO}DT0065_someT.dfy")
for i, why in [('DT0258', "BitwiseOr is defined as 0 but the task axiom BitwiseOrIdentity says BitwiseOr(x, 0) == x"),
               ('DT0333', "exp ≡ 1.0 and log ≡ 0.0 but LogExpProperties asserts log(exp(x)) == x"),
               ('DT0393', "cos ≡ 0.0 but cos_decreasing_property asserts cos(x) > cos(y) for 0 <= x < y <= PI")]:
    put(Dk(i), U, 'task_axiom_inconsistent', 'task',
        f"The task's {{:axiom}} lemmas contradict its own function bodies ({why}); the solution calls them, and "
        "`lemma ensures false` verifies in the published file. The proved postconditions are false for the defined functions.",
        f"{DEMO}{i}_sol_plus.dfy: verified, 0 errors")
put(Dk('DT0085'), B, 'task_axioms_consistent', 'task',
    "Spec includes task-provided {:axiom} lemmas about an extern BitwiseOr (x|0 = x, x|-1 = -1, commutative, "
    "associative, idempotent), all true of real bitwise OR, so consistent; solution uses them as given. Rung requires "
    "every frozen lemma to verify, which no solution can do.")
put(Dk('DT0285'), B, 'task_axioms_consistent', 'task',
    "Task {:axiom} lemmas about extern Sin/Arcsin are jointly satisfiable (e.g. Arcsin(x) = HALF_PI*x, Sin(y) = "
    "y/HALF_PI); solution uses them as given.")
put(Dk('DT0330'), B, 'task_stub_axioms_true', 'task',
    "Task stubs log ≡ 0.0; its {:axiom} lemmas (log(1) = 0, monotone) are true of the stub, so nothing false is used. "
    "Task is degenerate (log1p returns zeros) but the solution meets the spec as written.", f"{DEMO}DT0330_sol_plus.dfy")
put(Dk('DT0367'), B, 'task_stub_axioms_true', 'task',
    "Task stubs sin ≡ 0, cos ≡ 1, tan ≡ 0; axiom tan = sin/cos holds for the stubs. Degenerate task, consistent.",
    f"{DEMO}DT0367_sol_plus.dfy")
put(Dk('DS0048'), B, 'task_extra_axiom_method', 'task',
    "Task spec contains an extra bodiless `method {:axiom} ShapeArrays` besides the target ShapeMatrix; it is never "
    "called, and the solution `[a.m, a.n]` is correct. Rejected only because the task's bodiless method cannot verify "
    "or compile.")
put(Dk('DS0030'), B, 'task_bodiless_spec_fn', 'task',
    "Task preamble declares `function ShiftLeftInt(x: int, shift: nat): int` without body and the spec is "
    "result[i] == ShiftLeftInt(a[i], b[i]); calling it is the only provable implementation, which therefore cannot be "
    "compiled. Proof sound; no executable solution exists for this task.")
put(Dk('DV0164'), B, 'task_unused_bodiless_fn', 'task',
    "Code is a correct verified loop and never calls the task's unused bodiless ReplaceLoopSpec; `translate` fails only "
    "because that dead task declaration has no body.")
put(Dk('DV0125'), Q, 'task_axiom_fns_called', 'task',
    "Task preamble gives {:axiom} bodiless FirstEvenIndex/FirstOddIndex with complete (consistent) ensures; the solution "
    "calls them, so the proof is sound but the code cannot be compiled; an executable loop proven against those ensures "
    "was possible.")
put(Dk('DV0067'), B, 'vacuous_task_lemma', 'task',
    "The frozen spec is a lemma with requires only and no ensures; `{}` is the only possible proof. Rung flags it "
    "because Dafny emits no correctness result for an obligation-free lemma. The task (not the solution) is vacuous.")
put(Dk('DH0126'), B, 'task_decreases_star', 'task',
    "The frozen spec itself declares `decreases *` (Collatz: termination is an open problem), so the loop's "
    "`decreases *` is required; the rung bans the construct everywhere. Partial correctness is what the task asks.")
put(Dk('DD0164'), B, 'extra_decreases_clause', 'rung',
    "Model inserts `decreases n` between the spec and the body of the recursive method (a termination measure, sound "
    "and needed since x: real); the rung requires the body to follow the spec directly.")

# ------------------------------------------------------------------------------------------------ Verus (34 text hits)
Vk = lambda i: f'verus/{i}'
EB = {
 'VA0250': ('external_body returns String::from("0\\n") with trusted `ensures valid_output(result@)`; solve returns it', False),
 'VB0011': ('external_body `ensures v@ == s@` but the body is `Vec::new()`: the trusted contract is false', False),
 'VD0040': ('external_body int_to_i64(x: Ghost<int>) `ensures r == x@` with body unimplemented!(): false for |x| > i64::MAX', False),
 'VD0043': ('external_body str_len_exec `ensures result == s@.len()` returns s.len() (bytes), false for non-ASCII', False),
 'VD0094': ('external_body mul3 `ensures r == 3 * x` on i32 with body unimplemented!(): false on overflow', False),
 'VD0298': ('external_body returns `false` with `ensures found <==> have_common_k_substring_pred(...)`', False),
 'VD0425': ('external_body proof fn int_div_exists returns arbitrary() with the full postcondition; also turns the spec `fn` into `proof fn`', False),
 'VD0438': ('`#[verifier::external_body] proof fn axiom_one_eq_two() ensures 1 == 2` discharges the (impossible) task spec `ensures 1==2`', False),
 'VD0747': ('external_body compute_cylinder_area on u64 `ensures area == 2*r*(r+h)` ignoring overflow', False),
 'VD0772': ('external_body proof fn `ensures r*r <= i32::MAX` for every r > 0 (false for r = 50000)', False),
 'VD0776': ('external_body broadcast lemma: swapping two entries preserves the multiset (true, but unproven)', True),
 'VD0796': ('external_body usize_from_nat `ensures u == n@`, body unimplemented!(): false for n > usize::MAX', False),
 'VD0843': ('external_body proof fn asserts power(n, alpha) == 0 for the task\'s uninterpreted power; code returns 0', False),
 'VT0016': ('external_body exec wrapper that "calls" the spec fn f, body unimplemented!()', False),
 'VT0112': ('external_body get_nan `ensures result != result`, which is false in Verus\'s logic (spec equality is reflexive): asserts False', False),
 'VT0213': ('external_body constructor of PrintOptions whose ensures match its body (true but unverified)', True),
 'VT0266': ('external_body proof fn `ensures forall|x: f32| !is_nan_f32(x) && !is_infinite_f32(x)` (false)', False),
 'VT0313': ('external_body binds exec powf to the task\'s uninterpreted spec fn float_pow (the natural, consistent binding, but trusted)', True),
 'VT0543': ('external_body proof fn: ANY Vec of length n equals pcg64_dxsm_spec(seed, n), plus seed-injectivity of an arbitrary() spec fn: contradictory', False),
 'VV0036': ('external_body proof fn: ANY result is the majority element', False),
 'VV0047': ('external_body sort_vec returns its input unchanged with `ensures sorted`', False),
 'VV0065': ('spec has no postcondition; the whole body is an external_body Rust function (nothing verified)', False),
}
for i, (why, true_claim) in EB.items():
    put(Vk(i), U, 'external_body_trusted' if true_claim else 'external_body_false_contract', 'model',
        f"Proof relies on a model-written #[verifier::external_body] item: {why}." +
        (" The trusted claim is true, so this is unverified rather than wrong." if true_claim else ""),
        'text audit + reading (no Verus toolchain run)')
for i in ['VD0037', 'VD0042', 'VD0080', 'VD0091', 'VD0092', 'VD0093', 'VD0493', 'VD0703', 'VD0786', 'VD0873']:
    put(Vk(i), B, 'fn_to_proof_fn', 'task',
        "Only change to the frozen spec is `fn` -> `proof fn`: the record's exec signature takes/returns ghost `int`/Seq "
        "types, which Verus exec code cannot use, so a ghost function is the only way to state it; requires/ensures are "
        "unchanged and the body uses no assume/admit/external_body." +
        (" (The `assume(false)` in VD0037 is in the task's unrelated from_array stub, which is not called.)" if i == 'VD0037' else ''),
        'text diff + reading (no Verus toolchain run)')
put(Vk('VD0773'), U, 'spec_postconditions_deleted', 'model',
    "Frozen `fn smallest_missing_number(s) -> (v: int) requires ... ensures 0 <= v, !s.contains(v), forall k < v ==> "
    "s.contains(k)` became `spec fn smallest_missing_number(s) -> int recommends ... { 0 }`: all postconditions are "
    "gone and the body is the constant 0.", 'text diff')
put(Vk('VH0105'), U, 'exploit_weak_spec', 'model',
    "Preamble difference is benign (number_to_name moved from vc-helpers, identical, unused). But the spec only says "
    "result.len() <= arr.len() and every element is a digit name, and the code returns an empty Vec: nothing is "
    "sorted, reversed or named.", 'reading')

# ------------------------------------------------------------------------------------------------ emit
rows = []
tflags = {h['file']: h['hatches'] + h['tamper'] for h in TEXTHITS}
for r in TARGET:
    k = r['file'].rsplit('.', 1)[0].replace('_vericoded', '')
    key = k if r['lang'] != 'lean' else f"lean/{r['id']}/{r['model']}"
    verdict, mech, fault, why, ev = V[key]
    rows.append({'key': r['key'], 'id': r['id'], 'lang': r['lang'], 'source': r['source'], 'model': r['model'],
                 'file': r['file'], 'set': 'baseline_accept_sorryless_reject' + ('+text_hit' if r['file'] in tflags else ''),
                 'baseline_status': r['baseline_status'], 'sorryless_status': r['sorryless_status'],
                 'sorryless_violations': r['violations'][:3], 'text_flags': tflags.get(r['file'], []),
                 'verdict': verdict, 'mechanism': mech, 'fault': fault, 'why': why, 'evidence': ev})
for h in TEXTHITS:
    if h['lang'] != 'verus':
        assert any(x['file'] == h['file'] for x in rows), h
        continue
    verdict, mech, fault, why, ev = V[f"verus/{h['id']}"]
    rows.append({'key': h['file'], 'id': h['id'], 'lang': 'verus', 'source': h['source'], 'model': None,
                 'file': h['file'], 'set': 'text_hit', 'baseline_status': None, 'sorryless_status': None,
                 'sorryless_violations': [], 'text_flags': h['hatches'] + h['tamper'],
                 'verdict': verdict, 'mechanism': mech, 'fault': fault, 'why': why, 'evidence': ev})
assert len(rows) == len(V) == 249, (len(rows), len(V))
REPO_OUT.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

C = collections.Counter
out = {'n': len(rows),
       'by_verdict': C(r['verdict'] for r in rows),
       'by_lang_verdict': {f"{l}": dict(C(r['verdict'] for r in rows if r['lang'] == l)) for l in ('lean', 'dafny', 'verus')},
       'lean_by_model_verdict': {m: dict(C(r['verdict'] for r in rows if r['model'] == m))
                                 for m in sorted({r['model'] for r in rows if r['model']})},
       'by_fault_verdict': {f: dict(C(r['verdict'] for r in rows if r['fault'] == f)) for f in sorted({r['fault'] for r in rows})},
       'by_mechanism': {f"{r[0]}|{r[1]}|{r[2]}": n for r, n in
                        sorted(C((x['lang'], x['verdict'], x['mechanism']) for x in rows).items(), key=lambda t: (t[0][0], t[0][1], -t[1]))},
       'b_plus_s_minus_only': {l: dict(C(r['verdict'] for r in rows if r['lang'] == l and r['set'].startswith('baseline')))
                               for l in ('lean', 'dafny')}}
print(json.dumps(out, indent=1, default=dict))
