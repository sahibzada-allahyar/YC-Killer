"""Command line.

  python -m millennium_rsi run --solver z-ai/glm-5.3-flash --improver claude-haiku-5-5 --generations 4 \\
      --train 12 --val 12 --out runs/rsi/demo
  python -m millennium_rsi run ... --reward exit0          # ablation: optimise the usual exit-code criterion
  python -m millennium_rsi lemma path/to/lemma.lean        # would this lemma be admitted to the library?
  python -m millennium_rsi report runs/rsi/demo            # the self-improvement curve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import improve, tasks, verify


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="millennium_rsi", description="Recursive self-improvement with a verified reward.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--solver", default="z-ai/glm-5.3-flash")
    r.add_argument("--improver", default="claude-haiku-5-5")
    r.add_argument("--generations", type=int, default=4)
    r.add_argument("--train", type=int, default=12)
    r.add_argument("--val", type=int, default=12)
    r.add_argument("--rounds", type=int, default=2, help="solver attempts per task, with the reward's feedback")
    r.add_argument("--reward", choices=verify.REWARDS, default="millennium")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--out", type=Path, required=True)
    lm = sub.add_parser("lemma")
    lm.add_argument("path", type=Path)
    rp = sub.add_parser("report")
    rp.add_argument("out", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        train, val = tasks.load(a.train, a.val)
        s = improve.run(train, val, a.solver, a.improver, a.generations, a.out, verify.find_lean(), a.reward,
                        a.rounds, a.workers)
        print(json.dumps({k: s[k] for k in ("seed_val", "best_val", "best_harness")}, indent=1))
        return 0
    if a.cmd == "lemma":
        ok, why = verify.check_lemma(a.path.read_text(), verify.find_lean())
        print(("ADMITTED" if ok else "REJECTED") + f": {why}")
        return 0 if ok else 1
    s = json.loads((a.out / "summary.json").read_text())
    print(f"reward={s['reward']} solver={s['solver']} improver={s['improver']}")
    for c in s["curve"]:
        print(f"  gen {c['g']:>2}  {c['id']}  proved {c['proved']:>3}/{c['n']}  exit0 {c['exit0']:>3}  "
              f"fake {c['hacks']:>2}  ${c['cost']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
