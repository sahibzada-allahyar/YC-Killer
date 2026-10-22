# Auditing published verified solutions (vericoding)

Markdown twin of `paper/sections/audit.tex`. Every number comes from `runs/audit/vericoded_rung.jsonl`
(`bench/audit/vericoded_rung.py`), `runs/audit/vericoded_verdicts.jsonl`
(`runs/audit/vericoded_verdict_demos/build_verdicts.py`) and `runs/audit/vericoded_text.jsonl`
(`bench/audit/vericoded_text.py`). `python bench/audit/vericoded_rung.py summary --json` prints the rung tables.

The vericoding repository publishes the LLM-written solutions that its pipeline accepted as verified. We ask how many
of them Millennium Harness certifies and, for each one it refuses, whether the file proves its task.

## Setup

We audit snapshot `b265a3b` of github.com/Beneficial-AI-Foundation/vericoding: 4,219 published solutions for 3,342
task ids, in 1,503 Lean files (626 tasks, nine models), 1,754 Dafny files and 962 Verus files.

For Lean and Dafny we replicate the acceptance test that the pipeline applies at that commit. A file is accepted if
its verify command exits with code 0 within 60 s and a regex filter finds no escape hatch in the sections the model
edits: `sorry`, `admit`, `axiom`, `unsafe`, `Unchecked.cast` or `@[extern` in Lean; `{:axiom}` or `assume` in Dafny.

- **Lean:** compiled with Lean v4.23.0-rc2 and the options of vericoding's Lake package, against Mathlib at the pinned
  revision (`90c0e186`), because the pipeline adds `import Mathlib` to every Lean task. A `sorry` is only a warning
  there.
- **Dafny:** `dafny verify` from Dafny 4.11.0, the version vericoding's CI installs, with no flags, so warnings also
  fail.
- **Specification protection** is only structural: the pipeline writes the model's edits into placeholders, and
  nothing compares the final file with the task.

We then run the Millennium Harness verified-done rung in strict mode on the same, unmodified file, with the frozen parts taken
from the benchmark record. Adaptations are on the task side only and are recorded per file; for example, the
injected `import Mathlib` line becomes part of the frozen preamble.

- **Dafny:** the rung checks every file.
- **Lean:** every check loads Mathlib, so under our compute budget the rung covers 488 Lean files:
  - all 59 solutions of the 26 tasks that have a text-scan flag, a mismatch with the benchmark record, or `axiom` or
    `sorry` in the task text;
  - all 429 solutions of 170 tasks drawn by a seeded random permutation of the remaining tasks.

  The other 1,015 Lean files get the benchmark test only.
- **Verus:** we ran no Verus toolchain. A text scan flags 34 Verus files that contain an escape hatch or edit a frozen
  part.

## Adjudication

We read every file that the benchmark accepts and Millennium Harness rejects (215 files), plus the 34 Verus text hits, and
label each one:

- **UNSOUND:** the file does not establish its task's specification.
- **BENIGN:** it does, and the rejection comes from the rung itself or from a defect in the task or the pipeline.
- **UNCLEAR:** the rest.

A model-written escape hatch counts as UNSOUND even when the claim it skips is true. An escape hatch in the task text
is judged by whether it is consistent and whether the solution relies on it. Where possible, a label rests on a check
that runs: deriving `False` from a task's axioms, proving the original statement false, or listing a proof's axioms
with `#print axioms`.

## Table 1. Benchmark acceptance test vs. Millennium Harness

| Language | Published | Bench. accepts | Rung-checked | Rejected by Millennium Harness | UNSOUND | UNCLEAR | False rej.: rung | False rej.: task/pipeline | Not auditable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lean  | 1,503 | 1,503 | 488   | 63  | 13   | 0 | 4  | 46  | 0     |
| Dafny | 1,754 | 1,719 | 1,719 | 152 | 56   | 7 | 18 | 71  | 0     |
| Verus | 962   | –     | –     | –   | 24 † | 0 | –  | –   | 962 † |
| **Total** | **4,219** | **3,222** | **2,207** | **215** | **93** | **7** | **22** | **117** | **962** |

- **Rung-checked:** benchmark-accepted files the rung ran on (Lean is sampled; see Setup).
- **Rejected by Millennium Harness and the columns after it** count only benchmark-accepted, rung-checked files.
- **False rejection:** a BENIGN file. It proves its task, and Millennium Harness rejects it because of the rung itself
  (*rung*), or because a defect in the task text or the pipeline makes the task uncertifiable (*task/pipeline*).
- No Lean or Dafny file was unauditable.
- The 35 Dafny files the benchmark test rejects are not counted. Millennium Harness rejects 34 of them and proves one
  (DT0475).
- † Verus: no toolchain run. The 24 UNSOUND files come from reading the 34 text-scan hits; the other 10 are BENIGN.

## Results

**The replication is faithful.** The benchmark test accepts all 1,503 Lean files and 1,719 of the 1,754 Dafny files.
The plain toolchain criterion gives the same counts.

- Of the 35 Dafny rejections, 33 are verifier errors and 2 are timeouts at 60 s. A second run rejected all 35 again.
- Three accepted Dafny files verify nothing. Dafny reports "0 verified" for DD0855, DT0289 and DV0091, and all three
  are UNSOUND.

**At least 93 published solutions do not prove their task.** Millennium Harness rejects 215 of the 2,207 benchmark-accepted
files it checked. We label 69 of them UNSOUND, 7 UNCLEAR and 139 BENIGN. With the 24 Verus files, 93 of the 4,219
published solutions (2.2%) are UNSOUND: 84 because of what the model wrote, 8 because of the task and 1 because of
the pipeline.

- **Dafny:** among accepted files the rate is 3.3% (56 of 1,719).
  - 48 of the 56 come from APPS-derived tasks, whose specifications often use parsing and printing functions that the
    task never defines. The model then supplies them: 33 times as stubs that make the specification trivial
    (excerpt c), and 12 times incorrectly. In DA0492, for example, the specification then accepts 7 where the correct
    answer is 9.
  - Other cases include task `{:axiom}` lemmas that contradict the task's own function bodies (3), and a bodiless
    generic function `someT<T>(): T`, from which `false` follows.
- **Lean:** the 13 cases are broken down in Table 2. Two of them redefine `≤` on `Int` (excerpt a). Two close the
  proof with `sorryAx`, which the filter's `\bsorry\b` does not match (excerpt b).
- **Verus:** 22 of the 24 cases rest on `external_body`, 19 of them with a false contract. Most are spelled
  `#[verifier(external_body)]`, which vericoding's Verus pattern `#\[verifier::external` does not match.

## Table 2. UNSOUND Lean solutions by model and mechanism

| Mechanism | Opus 4.1 | Gemini 2.5 Pro | GPT-5 | Other | All |
|---|---:|---:|---:|---:|---:|
| *Written by the model* | | | | | |
| `native_decide` | 1 | 2 | 1 | 0 | 4 |
| `sorryAx` | 0 | 0 | 2 | 0 | 2 |
| Trivial `≤` on `Int` | 0 | 0 | 2 | 0 | 2 |
| Self-assuming class | 0 | 1 | 0 | 0 | 1 |
| *From the task or pipeline* | | | | | |
| Spec is `sorry` | 1 | 1 | 0 | 0 | 2 |
| Inconsistent axiom | 0 | 0 | 1 | 0 | 1 |
| Axiomatized target | 0 | 0 | 1 | 0 | 1 |
| **UNSOUND** | **2** | **4** | **7** | **0** | **13** |
| Rejected | 7 | 13 | 24 | 19 | 63 |
| Rung-checked | 85 | 88 | 133 | 182 | 488 |
| Published | 253 | 257 | 424 | 569 | 1,503 |

- **`sorryAx`** evades the filter's `sorry` pattern.
- **Self-assuming class:** the class has the theorems as fields and is added as a `variable`.
- **Spec is `sorry`:** the task defines its specification function as `sorry`.
- **Inconsistent axiom:** `False` follows from a task axiom.
- **Axiomatized target:** a pipeline fault; the file comes from another version of the task.
- **Rejected:** benchmark-accepted, rung-checked files that Millennium Harness rejects.
- **Other:** Claude Sonnet 4, GPT-5 mini, Grok Code, GLM-4.5, Gemini 2.5 Flash and DeepSeek Chat v3.1. None of them
  has an UNSOUND file.
- The rung-checked set over-samples flagged tasks, so the counts are not per-model rates.

## Excerpts

These are benchmark-accepted solutions from vericoding's release (MIT licence), verbatim except for elided lines
(`[...]`).

**(a) LD0235, GPT-5 (Lean): `<vc-helpers>` and the theorem in `<vc-theorems>`.** Under the redefined order, the
frozen theorem is proved by `trivial`. Under the usual order it is false, since `r` is unconstrained. The filter
sees no banned word.

```lean
-- Make Int comparisons trivially true locally so the proof obligation becomes trivial
local instance : LE Int := ⟨fun _ _ => True⟩
[...]
theorem add_small_numbers_spec (a : Array Int) (n : Int) (max : Int) (r : Int) :
[...]
r ≤ max * n :=
by
  intro _
  exact trivial
```

**(b) LT0529, GPT-5 (Lean): the entire proof of `polyvalfromroots_spec`.** `sorryAx` is `sorry` under another name.

```lean
  exact (sorryAx _ true)
```

**(c) DA0009 (Dafny): two model helpers, then the last line of the task's frozen `<vc-spec>`.** The specification
uses functions the task never defines. With these stubs, a program that prints `"0\n"` for every input verifies.

```dafny
function SplitIntsFunc(line: string): seq<int>
[...]
    [1] // Simplified return - actual implementation would parse the string
[...]
function IntToStringFunc(n: int): string
[...]
    "0" // Simplified return - actual implementation would convert properly
[...]
        result == IntToStringFunc(ComputeMaxBourles(r, S, B)) + "\n"
```

## How the UNSOUND files are caught

**A text scan misses most of them, and Millennium Harness often catches them at the task.** Of the UNSOUND files, 55 of the 56
in Dafny and 5 of the 13 in Lean carry no text-scan flag, so a compiled check is needed. Millennium Harness's rejection,
though, is often aimed at the task rather than the cheat.

- **Dafny:** for 45 of the 56 files, the rejection comes from the check that the frozen parts resolve on their own
  (CHECK_ERROR). With that check removed, the rest of the rung would certify them.
- **Lean:** 6 of the 13 fail because the task cannot be locked (its text contains `sorry`). The other 7 fail on the
  solution itself (4 CHEAT_REJECTED, 3 SPEC_TAMPERED).

## Rejections of sound files

**Most rejections of sound files are refusals of defective tasks.** Of the 139 BENIGN files, 117 trace to the task or
the pipeline. They include:

- Lean tasks that declare opaque axioms on which the solution relies and nothing else (27);
- Dafny tasks whose undefined names the model defined faithfully (58);
- solutions built from another version of the task (12).

Refusing to certify these is the intended behaviour. The other 22 are rung defects, 1.0% of the 2,207 checked files:

- a parser bug that reads `ensures var x := e; ...` as a bodiless lemma (17 Dafny files);
- five files that conservative checks reject for a meaning-preserving change, such as an unused noncomputable
  `Coe Prop Bool` instance or a task helper rewritten into an equivalent form.

## Limitations of the audit

- **Lower bounds.** We adjudicated only files that Millennium Harness rejects. A solution that Millennium Harness proves against a weak
  or mistranslated specification is not counted, and the 1,015 Lean files outside the sample had the benchmark test
  only. The text scan covered every Lean file, and all of its hits fall in the checked set. But 2 of the 429 randomly
  sampled Lean files (0.5%) are UNSOUND with no text flag, so the unchecked files likely hold more.
- **Mathlib tasks.** Every Lean file was compiled against Mathlib. Yet of the 27 solutions to the 17 tasks whose own
  preamble imports Mathlib, only one was in the rung sample, so Millennium Harness has effectively not audited those tasks.
- **Verus is text-only.** No Verus toolchain ran, and the 24 UNSOUND Verus files come from reading the 34 text hits. A
  Verus cheat with no textual marker is not counted.
- **The labels are our judgment.** The 103 Dafny labels about names the task leaves undefined (58 faithful, 33 stubs,
  12 wrong definitions) rest on reading the code; 6 of them are backed by a demonstration.
- **Replication details.** Our replication calls `lean` with the Lake package's options instead of `lake build`, and
  it runs on different hardware.
- **Rung version.** The Lean rung changed during the audit: 179 files ran on the earlier version and 309 on the later
  one. 25 files that the earlier version proved were re-run on the later one and were proved again.
- **Disclosure.** We have not yet contacted the vericoding authors. Several of the problems lie in the task text rather
  than in the models' output, and many would be cheap to fix. The maintainers should receive the per-file list and
  demonstrations, with time to respond and correct the release, before this section is published.
