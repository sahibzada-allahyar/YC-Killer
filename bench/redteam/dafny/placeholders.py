"""vericoding's APPS-Dafny `vc-code` is a placeholder, not a proof: does plain `dafny verify` accept it?

  MILLENNIUM_DAFNY=~/.local/dafny-4.11/dafny/dafny python bench/redteam/dafny/placeholders.py [runs/dafny/vericoding.jsonl]

For every Dafny record with source `apps`, qa-issue 0 and vc-code `assume {:axiom} false`, the published file is
assembled exactly as bench/adapters/vericoding.py builds its oracle (preamble, helpers between the HELPERS markers,
spec, vc-code, postamble) and graded twice: plain `dafny verify --allow-warnings` (exit 0 = accepted) and
millennium_harness.dafny.check against the frozen preamble/spec/postamble. Prints a JSON summary; per-task rows go to
runs/redteam/dafny_placeholders.jsonl.
"""
from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from millennium_harness import dafny  # noqa: E402
from millennium_harness.dafny import HELPERS_BEGIN, HELPERS_END  # noqa: E402

PLACEHOLDER = re.compile(r"\A\{\s*assume\s*\{:axiom\}\s*false;\s*\}\Z")


def assemble(r: dict) -> str:
    pre, spec, post = r.get("vc-preamble") or "", r["vc-spec"], r.get("vc-postamble") or ""
    return (f"{pre}\n\n{HELPERS_BEGIN}\n{r.get('vc-helpers') or ''}\n{HELPERS_END}\n\n{spec}\n{r['vc-code']}\n\n"
            f"{post}\n")


def grade(r: dict) -> dict:
    text = assemble(r)
    with tempfile.TemporaryDirectory(prefix="sl-ph-") as d:
        Path(d, "solution.dfy").write_text(text, encoding="utf-8")
        exe = os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY", "dafny"))
        try:
            p = subprocess.run([exe, "verify", "--allow-warnings",
                                "solution.dfy"], cwd=d, capture_output=True, text=True, timeout=300)
            plain = p.returncode == 0
            m = re.search(r"finished with (\d+) verified, (\d+) errors?", p.stdout + p.stderr)
            summary = m.group(0) if m else (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]
        except subprocess.TimeoutExpired:
            plain, summary = False, "timeout"
    frozen = {"preamble": r.get("vc-preamble") or "", "spec": r["vc-spec"], "postamble": r.get("vc-postamble") or ""}
    s = dafny.check(text, frozen, "solution.dfy")
    return {"id": r["id"], "plain_accepts": plain, "plain_summary": summary if isinstance(summary, str) else summary[0],
            "millennium": s["status"], "violations": s["violations"][:3]}


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs" / "dafny" / "vericoding.jsonl")
    recs = [json.loads(ln) for ln in src.open(encoding="utf-8")]
    apps = [r for r in recs if r["language"] == "dafny" and r["source"] == "apps"]
    clean = [r for r in apps if r.get("qa-issue") == 0]
    ph = [r for r in clean if PLACEHOLDER.match((r.get("vc-code") or "").strip())]
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        rows = list(ex.map(grade, ph))
    out = ROOT / "runs" / "redteam"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dafny_placeholders.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({
        "dafny_apps_records": len(apps), "qa_clean": len(clean), "placeholder_vc_code": len(ph),
        "plain_dafny_verify_accepts": sum(r["plain_accepts"] for r in rows),
        "millennium_status": dict(collections.Counter(r["millennium"] for r in rows)),
        "millennium_status_among_plain_accepted": dict(collections.Counter(r["millennium"] for r in rows
                                                                           if r["plain_accepts"])),
    }, indent=1))


if __name__ == "__main__":
    main()
