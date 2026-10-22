"""Per-case red-team tables for the paper appendix, from runs/redteam/*.jsonl (no toolchain needed).

  python bench/redteam/paper_tables.py > paper/sections/redteam_cases.tex

Inputs: runs/redteam/lean_baseline.jsonl (bench/redteam/lean/baseline.py), runs/redteam/lean_baselines.jsonl
(bench/baselines/lean_baselines.py), runs/redteam/dafny_baselines.jsonl (bench/baselines/dafny_baselines.py) and a fresh Millennium Harness run of bench/redteam/dafny/cases.json (bench/redteam/dafny/run.py
logic, imported). Families are fixed here and checked to cover every case exactly once.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs" / "redteam"

LEAN = {
    "Escape hatch": "sorry_proof admit sorryax_synthetic decreasing_by_sorry axiom_added native_decide decide_native "
                    "noncomputable_choose",
    "Edited, hidden or cut off": "edited_statement postamble_edit trailing_code guard_msgs_wrap private_theorem "
                                 "namespace_shadow comment_swap exit unclosed_comment",
    "Same text, new meaning": "instance_shadow_le variable_include_false local_notation macro_rules_theorem_to_axiom "
                              "command_elab_attr",
    "Tests hijacked by an instance": "display_instance local_instance_display ofnat_input_shift rt2_elab_1 rt2_elab_2 "
                                     "rt2_elab_3 rt2_elab_4 rt2_elab_5 rt2_elab_6 rt2_elab_7 rt2_elab_8 rt2_elab_9 "
                                     "def_alias_local_reducible_repr def_of_class_local_reducible_repr "
                                     "class_extends_repr generic_repr_via_agent_class scoped_abbrev_of_class_ofnat "
                                     "coe_bridge_through_own_type coetail_bridge_through_own_type "
                                     "one_reprioritized_toOfNat1 one_local_toOfNat1 coe_bridge_two_hops "
                                     "default_instance_library",
    "Compiled/meta code, kernel, lexer": "implemented_by extern_attr initialize_io module_header skip_kernel_tc "
                                         "run_cmd_addDecl run_tac_io elab_tactic macro_tactic by_elab_io "
                                         "guillemet_comment_trick char_literal_trick implemented_by_guillemet "
                                         "skip_kernel_tc_guillemet",
}
DAFNY = {
    "Escape hatch": "assume_false assume_axiom_placeholder axiom_lemma at_axiom_lemma at_space_axiom glued_at_axiom "
                    "verify_false_helper verify_false_paren bodiless_lemma bodiless_lemma_brace extern_method "
                    "decreases_star expect_postcondition compile_false_helper selective_checking_helper only_attribute "
                    "at_comment_resource_limit include_axioms refines_module cheat_before_preamble",
    "Not an implementation": "nocompile_such_that bodiless_loop bodiless_forall",
    "Spec edited or reinterpreted": "ghost_prefix_spec spec_in_verbatim_string line_comment_swallows_header "
                                    "abstract_module_wrap requires_false_after_spec weaken_ensures_after_spec "
                                    "spec_in_comment edited_spec class_shadow_predicate datatype_member_shadow "
                                    "autocontracts_false_valid function_shadows_constructor agent_defines_spec_helper "
                                    "agent_defines_spec_helper_vacuous",
}
SHORT = {"PROVED": "P", "UNPROVED": "U", "CHEAT_REJECTED": "C", "SPEC_TAMPERED": "T", "CHECK_ERROR": "E"}


def tt(s: str) -> str:
    return "\\texttt{" + s.replace("_", "\\_") + "}"


def yn(b: bool) -> str:
    return "\\ding{51}" if b else "--"


def family_of(fams: dict, name: str) -> str:
    hits = [f for f, v in fams.items() if name in v.split()]
    assert len(hits) == 1, (name, hits)
    return hits[0]


def lean_table() -> list[str]:
    rows = [json.loads(ln) for ln in open(R / "lean_baseline.jsonl", encoding="utf-8")]
    comp = {r["file"]: r["a_plus_d"] for r in map(json.loads, open(R / "lean_baselines.jsonl", encoding="utf-8"))}
    cheats = [r for r in rows if r["kind"] == "cheat"]
    honest = [r for r in rows if r["kind"] == "honest"]
    assert {Path(r["file"]).stem for r in cheats} == set(sum((v.split() for v in LEAN.values()), []))
    out = ["\\begin{table*}[p]", "\\centering", "\\scriptsize", "\\renewcommand{\\arraystretch}{0.93}",
           "\\begin{tabular}{@{}llcccccc@{}}", "\\toprule",
           "\\textbf{Lean case} & \\textbf{Family} & \\textbf{L0} & \\textbf{L1} & \\textbf{L2} & \\textbf{comp.+L2} & "
           "\\textbf{\\sys} & "
           "\\textbf{kernel-only} \\\\", "\\midrule"]
    for fam, names in LEAN.items():
        for n in names.split():
            r = next(x for x in cheats if Path(x["file"]).stem == n)
            out.append(f"{tt(n)} & {fam} & {yn(r['L0'])} & {yn(r['L1'])} & {yn(r['L2'])} & {yn(comp[r['file']])} & "
                       f"{SHORT[r['SL']]} & "
                       f"{SHORT[r['SL_lenient']]} \\\\")
    out.append("\\midrule")
    for r in honest:
        out.append(f"{tt(Path(r['file']).stem)} & honest & {yn(r['L0'])} & {yn(r['L1'])} & {yn(r['L2'])} & "
                   f"{yn(comp[r['file']])} & {SHORT[r['SL']]} & {SHORT[r['SL_lenient']]} \\\\")
    out += ["\\bottomrule", "\\end{tabular}",
            "\\caption{Lean red team, per case. \\ding{51}: the plain-Lean criterion accepts the file (L0 no error; L1 "
            "also no \\texttt{sorry} warning; L2 also \\texttt{\\#print axioms} $\\subseteq$ standard for every frozen "
            "theorem; comp.+L2 also the Lean FRO's comparator, the strongest existing check). \\sys statuses: P \\PROVED, U \\UNPROVED, C \\CHEAT, T \\TAMPERED; kernel-only is the lenient "
            "mode without the text ban scan. \\texttt{rt2\\_elab\\_9} is a control whose instance is inert; it fails "
            "the tests and is \\UNPROVED.}",
            "\\label{tab:lean-cases}", "\\end{table*}"]
    return out


def dafny_table() -> list[str]:
    base = {Path(r["file"]).stem: r for r in map(json.loads, open(R / "dafny_baselines.jsonl", encoding="utf-8"))}
    sl = {Path(r["file"]).stem: r for r in map(json.loads, open(R / "dafny_millennium.jsonl", encoding="utf-8"))}
    cheats = [n for n, r in base.items() if r["kind"] == "cheat"]
    assert set(cheats) == set(sum((v.split() for v in DAFNY.values()), []))
    out = ["\\begin{table*}[p]", "\\centering", "\\scriptsize", "\\begin{tabular}{@{}llcccl@{}}", "\\toprule",
           "\\textbf{Dafny case} & \\textbf{Family} & \\textbf{default} & \\textbf{allow-warn.} & \\textbf{all} & "
           "\\textbf{\\sys (first reason)} \\\\", "\\midrule"]

    def reason(n):
        v = (sl[n]["violations"] or [""])[0]
        v = v.split(" (allowed:")[0].split(": the frozen parts")[0].split(", so no solution")[0]
        v = v.split(". Possible mitigation")[0]
        return v[:56].rstrip().replace("\\", "").replace("_", "\\_").replace("{", "\\{").replace("}", "\\}") \
            .replace("`", "").replace("#", "\\#").replace("&", "\\&").replace("$", "\\$").replace("%", "\\%").replace("--", "-{}-")

    for fam, names in DAFNY.items():
        for n in names.split():
            b = base[n]
            short = {"Not an implementation": "Not an impl.", "Spec edited or reinterpreted": "Spec edited"}.get(fam, fam)
            out.append(f"{tt(n)} & {short} & {yn(b['D0'])} & {yn(b['D1'])} & {yn(b['accept']['f'])} & "
                       f"{SHORT[sl[n]['status']]}: {reason(n)} \\\\")
    out.append("\\midrule")
    for n, b in base.items():
        if b["kind"] != "cheat":
            kind = "honest" if b["kind"] == "honest" else "malformed task"
            extra = "" if b["kind"] == "honest" else f": {reason(n)}"
            out.append(f"{tt(n)} & {kind} & {yn(b['D0'])} & {yn(b['D1'])} & {yn(b['accept']['f'])} & "
                       f"{SHORT[sl[n]['status']]}{extra} \\\\")
    out += ["\\bottomrule", "\\end{tabular}",
            "\\caption{Dafny red team, per case. \\ding{51}: \\texttt{dafny verify} exits 0 (default warning mode, or "
            "with \\texttt{-{}-allow-warnings}); all: every existing check combined (\\texttt{dafny audit}, frozen-text diff, "
            "vericoding's pipeline). \\sys statuses as in Table~\\ref{tab:lean-cases}, plus E \\CHECKERR; "
            "the first violation is shown, abridged.}",
            "\\label{tab:dafny-cases}", "\\end{table*}"]
    return out


def main():
    if not (R / "dafny_millennium.jsonl").exists() or os.environ.get("RERUN_DAFNY"):
        sys.path.insert(0, str(ROOT / "bench" / "redteam" / "dafny"))
        import run as dafny_run  # noqa: E402
        with open(R / "dafny_millennium.jsonl", "w", encoding="utf-8") as f:
            for r in dafny_run.run_all():
                f.write(json.dumps({k: r[k] for k in ("file", "status", "violations", "raw_accepts", "ok")}) + "\n")
    print("% Generated by bench/redteam/paper_tables.py from runs/redteam/*.jsonl. Do not edit by hand.")
    print("\\section{Red-Team Cases}\n\\label{sec:appendix-redteam}\n"
          "Tables~\\ref{tab:lean-cases} and~\\ref{tab:dafny-cases} list every red-team case. They are reproduced by "
          "\\texttt{bench/redteam/lean/baseline.py}, \\texttt{bench/redteam/dafny/run.py}, "
          "\\texttt{bench/baselines/*.py} (the existing checkers) and \\texttt{bench/redteam/dafny/placeholders.py} (the "
          "vericoding placeholder audit), with Dafny 4.11 and Lean v4.23.0-rc2; "
          "\\texttt{tests/test\\_lean\\_rung.py} and \\texttt{tests/test\\_dafny\\_rung.py} run the same cases "
          "as tests.\n")
    print("\n".join(lean_table() + [""] + dafny_table()))


if __name__ == "__main__":
    main()
