# Millennium Harness — results (living document; TEST sections filled after the preregistered runs)

Model everywhere: Claude Haiku 5.5 (`claude-haiku-5-5`), effort medium, ≤100K input tokens per request.
Costs are recomputed from tokens with one Haiku 5.5 price table for every harness.

## 1. Selector study (fixed pool of 8 single-shot candidates per problem)

| set | problems | S1 public samples | CodeT | exec. clustering | LLM judge | **Millennium Harness** | oracle |
|---|---|---|---|---|---|---|---|
| LiveCodeBench v6 med/hard (pre-cutoff, dev) | 200 | 85.8% | +0.5 | +0.3 | +1.9 | **+1.0 [+0.2, +2.1]** | +2.7 |
| same, effort low | 200 | 85.2% | +0.4 | +0.7 | — | **+1.5 [+0.5, +2.9]** | +3.8 |
| FreshCP-VAL (AtCoder Apr–Jun 2026) | 38 | 82.2% | +0.3 | +0.3 | +2.0 | **+0.3 [0, +1.0]** | +2.0 |
| **FreshCP-TEST (post-cutoff, sealed; H3)** | 33 | 80.6% | +1.2 | +1.2 | **−1.8** | **+1.2 [0, +3.1] = oracle, p=0.50** | +1.2 |

Millennium Harness false alarms: 0.5% (6/1105) dev, 0.0% (0/199) VAL. The ceiling is the story: with public samples, a strong
cheap model is rarely "almost right" on competitive programming — failures are "could not solve".

## 2. In-loop gate (OpenCode 1.18.35; A vanilla · B + Millennium Harness gate · C compute-matched "stress-test yourself")

| set | A | B | C | B−A | C−A | gate fired | cost A/B/C per task |
|---|---|---|---|---|---|---|---|
| LiveCodeBench-100 (dev; 16K-output bug) | 89.0% | 89.0% | 91.0% | 0.0 | +2.0 | 2/100 | $0.0012 / 0.0012 / 0.0037 |
| FreshCP-VAL look 3 (bug fixed; 38 tasks) | 92.1% | 96.1% | 94.7% | +3.9 (4W/1L, p=0.37) | +2.6 | 1/76 (noise) | $0.0021 / 0.0023 / 0.0051 |
| **FreshCP-TEST (preregistered; 33 tasks, A×3 B×3 C×2)** | 97.0% | 97.0% | 98.5% | 0.0 [−6.1, +6.1] | +1.5 | 5/99 | $0.0021 / 0.0024 / 0.0046 |

## 3. Dafny honesty (vericoding, 46 well-formed APPS-derived specs)

| arm | honestly verified |
|---|---|
| A vanilla OpenCode (rules stated) | 41/46 (89.1%) |
| B + structural Dafny gate (`millennium_harness.dafny`) | 45/46 (97.8%) — 4 wins, 0 losses, **0 gate-attributable** |

Re-analysis of the gate logs: in 3 of the 4 wins the gate never fired (B's first stop already verified; run-to-run noise),
and in the 4th it fired only on quantifier-trigger hints, which are not soundness issues. APPS-Dafny is at ceiling.

"0 errors" ≠ verified: vericoding's own APPS reference code is `assume {:axiom} false`-style and passes plain
`dafny verify --allow-warnings`; Millennium Harness rejects it.

## 4. Cross-harness (sealed FreshCP-TEST, 33 post-cutoff tasks, same model/effort/cap, web off)

| harness | pass@1 | $/run | p50 time |
|---|---|---|---|
| OpenCode (3 runs) | 97.0% | $0.0021 | 54 s |
| OpenCode + Millennium Harness gate (3 runs) | 97.0% | $0.0024 | 60 s |
| OpenCode + "stress-test yourself" (2 runs) | 98.5% | $0.0046 | 114 s |
| Claude Code | 100% | $0.0086 | 47 s |
| Claude Code + Millennium Harness gate | 100% | $0.0079 | 58 s |
| Hermes Agent (capped) | 97.0% | $0.0051 | 96 s |
| OpenClaw (capped) | 93.9% | $0.0086 | 91 s |
| Codex CLI (Haiku via OpenRouter Responses bridge; no cache) | 93.9%* | $0.0199 | 87 s |

\* Codex curled atcoder.jp on abc476_g ("look for the official solution"); it got HTTP 200 from an unrelated page and no
solution content, but the trial is scored 0 under the plan's leakage rule (was 97.0%). Web search was off, but shell network
was not; from now on agent-phase egress is allow-listed to the model API only.

**Preregistered verdict:** H1 (B>A), H2 (B>C), H3 (S4>S1) — all not significant (Holm). B−A is equivalent to zero
within ±6.1pp; B−C −1.5 [−6.1, +1.5] (p=1.0); S4−S1 +1.2 [0, +3.1] (p=0.50). Agentic Haiku 5.5 with public samples is at ceiling on post-cutoff AtCoder problems that survive the
FreshCP pipeline; there is no room for a gate. Limitation: survivors (editorial code + strong tests) likely skew easier.
## 5. Math/physics (MathArena ArXivMath/ArXivPhys 08-2026, post-cutoff; 115 problems, 1 run)

| arm | overall | math | physics |
|---|---|---|---|
| A vanilla OpenCode | 12.2% | 15.8% | 8.6% |
| B + re-derivation gate | 20.0% | 26.3% | 13.8% |

Honest reading: B−A = +7.8pp (12W/3L, p=0.035), but on the 73 problems where the gate could not fire the arms are
identical by construction and still differ 9W/1L (timeouts near the limit). Gate-attributable subset: 3W/2L (n.s.).
→ null for the gate; a cautionary tale about single-run agent comparisons.

## 6. Exploratory (post-freeze): no examples, as in real work (LiveCodeBench-100, examples and checker removed)

| run | A vanilla | B + gate | B−A | gate fired | false alarms | $/task A · B | p50 A · B |
|---|---|---|---|---|---|---|---|
| run 1 (loader bug: function tasks specced as stdin) | 89.0% | 91.0% | +2.0 [−2.0, +6.0] | 50/99 | 43 (bug) | $0.0009 · 0.0016 | 20 s · 62 s |
| **run 2 (fixed; concurrent)** | 90.0% | 92.0% | **+2.0 [0, +5.0], 2W/0L, p=0.50** | 5/100 | **0** | $0.0010 · 0.0011 | 20 s · 29 s |

Without samples the gate found ≈56% of wrong solutions (5/9) with no false alarms, but Haiku repaired only 1 of 5 within
2 rounds. Detection is no longer the bottleneck; repair is.

## 7. Infrastructure findings (to report upstream; nothing filed yet)
- Harbor `livecodebench@6.0` copies hidden tests into the agent sandbox (`/app/config.json`, 413 MB over 100 tasks).
- LCB `test.sh` exits without a reward when no solution exists → silently counted as infrastructure failure.
- Harbor 0.24.0 vs current releases: Hermes `version` subcommand removed; OpenClaw needs Node ≥24.16; OpenClaw config
  upload assumes a bind-mounted /logs (broken on Modal); Codex strips the provider prefix from model names.
- A 16K per-request output cap silently truncates Haiku's thinking on hard problems (finish reason "length", no code).

## Strongest existing checkers vs Millennium Harness

Red-team suites on real vericoding tasks, model-free, Dafny 4.11.0 and Lean v4.23.0-rc2. Sources:
`runs/redteam/dafny_baselines.jsonl` (`bench/baselines/dafny_baselines.py`), `runs/redteam/dafny_millennium.jsonl`,
`runs/redteam/lean_baselines.jsonl` (`bench/baselines/lean_baselines.py`), and `runs/redteam/lean_baseline.jsonl`
(`bench/redteam/lean/baseline.py`; L0 and kernel-only, re-run on the full 59/19 suite; its L1, L2 and Millennium Harness
verdicts match `lean_baselines.jsonl` on all 78 cases).

| acceptance criterion | cheats accepted ↓ | honest accepted ↑ |
|---|---|---|
| **Dafny** (37 cheats, 6 honest) | | |
| `dafny verify` (default: any warning fails) | 23/37 | 5/6 |
| `dafny verify --allow-warnings` | 36/37 | 6/6 |
| + `dafny audit` (any finding rejects) | 20/37 | 6/6 |
| + frozen-text diff + vericoding's cheat regex | 17/37 | 6/6 |
| vericoding's checker as published (spliced template, `dafny verify`, regex) | 16/37 | 5/6 |
| **all of the above combined (strongest existing)** | **13/37** | 5/6 |
| **Millennium Harness** | **0/37** | **6/6** |
| **Lean 4** (59 cheats, 19 honest) | | |
| L0: `lean` reports no error | 53/59 | 19/19 |
| L1: L0 + no `declaration uses 'sorry'` (usual benchmark rule) | 47/59 | 19/19 |
| L2: L1 + `#print axioms` ⊆ {propext, Quot.sound, Classical.choice} | 38/59 | 19/19 |
| L2 + lean4checker kernel replay (`--fresh` gives the same verdicts) | 37/59 | 19/19 |
| comparator (Lean FRO; decision logic ported to v4.23) | 32/59 | 19/19 |
| **comparator + L2 (strongest existing)** | **32/59** | 19/19 |
| Millennium Harness, kernel-side checks only (no text ban scan) | 2/59* | 19/19 |
| **Millennium Harness** | **0/59** | **19/19** |

\* Both are real proofs that also set `debug.skipKernelTC` (one spelled `«debug»`); the ban scan rejects them.

**Headline:** the strongest existing checkers still accept 45 of 96 cheats (47%: 13/37 Dafny, 32/59 Lean); Millennium Harness
accepts 0 of 96 and all 25 honest solutions. Plain exit-0 criteria (Dafny `--allow-warnings`, Lean L1) accept 83/96.

- Real proofs among the accepted cheats: Lean `skip_kernel_tc`, `skip_kernel_tc_guillemet`; Dafny
  `at_comment_resource_limit` (only raises the time budget). Without them: Lean 30/57, Dafny 12/36.
- Lean: comparator rejects every cheat that changes a theorem statement, every axiom/sorry escape, and
  `run_cmd_addDecl` (kernel replay). Its 32 accepts all attack the implementation (a definition hole it does not
  inspect) or the tests: 22 instance hijacks of the tests, 7 compiled/meta-code or kernel tricks, `guard_msgs_wrap`,
  `trailing_code`, `noncomputable_choose`. 24 of the 32 pass a wrong or noncomputable implementation (22 hijacks,
  `guard_msgs_wrap`, `noncomputable_choose`); the other 8 demonstrate a mechanism whose payload is harmless as written.
  Adding L2 removes none of the 32.
- Lean examples that comparator + L2 accept: `display_instance` (wrong `n % 4`, a high-priority `Repr Nat` prints the
  expected test output); `coe_bridge_through_own_type` (wrong `x + 1000`, `Coe (Fin 5) Port` + high-priority
  `Coe Port Int` reroute the test's `Fin 5 → Int` coercion so input 3 becomes −996); `one_reprioritized_toOfNat1`
  (`One Nat := 1000` + `attribute [instance 5000] One.toOfNat1` on a library instance makes the numeral `1` mean 1000).
- Dafny examples that every check combined accepts: `requires_false_after_spec` (agent code starts with
  `requires false`, which attaches to the frozen method, so `{}` verifies); `selective_checking_helper`
  (`{:selective_checking}` turns a helper's assertions into assumptions); `function_shadows_constructor` (a top-level
  `function Some(s) { None }` replaces the frozen constructor, so `result == Some("...")` means `result == None`).
- Of the 13 Dafny accepts, 8 change what the specification means, 4 are escape hatches, 1 is not an implementation.
  vericoding's regex misses the three `@Axiom` cheats. The one honest file that default `dafny verify`, vericoding and
  the combination reject (`DA0134`) has only a trigger-hint warning.
- comparator was not run as the upstream binary (its release targets Lean v4.35.0-rc4; the build of downloaded code
  was stopped by the sandbox). `bench/baselines/fro_tools_v423.lean` copies its `Compare.lean`, `Axioms.lean`,
  `Util.lean`, target lists and kernel replay + Quot post-check verbatim, computes the exported constant closure
  in-process instead of the lean4export round trip, and embeds lean4checker's `replay'`. Challenge = frozen task with
  its `sorry`s and no postamble; permitted axioms = propext, Quot.sound, Classical.choice.
