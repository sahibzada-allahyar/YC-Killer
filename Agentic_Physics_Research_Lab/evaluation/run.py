from __future__ import annotations
import argparse, yaml, os, json
from evaluation.metrics import has_min_plots, posterior_widths_ok, code_executed_ok, citations_present

def run_suite(suite_path: str) -> int:
    spec = yaml.safe_load(open(suite_path))
    suite = spec["suite"]
    failures = 0
    for task in spec["tasks"]:
        tid = task["id"]
        base = os.path.join("artifacts", tid)  # your pipeline writes here
        expect = task.get("expect", {})
        ok = True
        if "plots_min" in expect:
            ok &= has_min_plots(os.path.join(base, "Plots"), expect["plots_min"])
        if "posterior_targets" in expect:
            ok &= posterior_widths_ok(os.path.join(base, "posteriors.json"),
                                      expect["posterior_targets"])
        if "code_runs" in expect:
            logp = os.path.join(base, "logs.txt")
            ok &= os.path.isfile(logp) and code_executed_ok(open(logp).read())
        if "citations_ok" in expect:
            mdp = os.path.join(base, "paper", "results.md")
            ok &= os.path.isfile(mdp) and citations_present(open(mdp).read())

        print(f"[{suite}:{tid}] -> {'OK' if ok else 'FAIL'}")
        failures += (0 if ok else 1)
    return failures

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    help="Path to evaluation suite YAML")
    args = ap.parse_args()
    exit(run_suite(args.suite))
