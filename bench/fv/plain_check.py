"""For every FV trial whose first stop (S0) is not PROVED by Millennium Harness, ask the plain toolchain: would the
usual acceptance criterion ('dafny verify' with 0 errors, or 'lean' with no errors; sorry is only a warning) have
counted it as verified? Local toolchains; DEV probe jobs only.  python bench/fv/plain_check.py runs/jobs/<job> [...]"""
import gzip
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

LEAN = os.path.expanduser("~/.elan/toolchains/leanprover--lean4---v4.23.0-rc2/bin/lean")
DAFNY = os.environ.get("MILLENNIUM_DAFNY",
                       os.environ.get("SORRYLESS_DAFNY", os.path.expanduser("~/.local/dafny-4.11/dafny/dafny")))


def s0_solution(trial: Path):
    f = trial / "agent" / "fv" / "snap" / "S0.tar.gz"
    if not f.exists():
        return None, None
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(f.read_bytes()))) as t:
        for m in t.getmembers():
            if m.name.endswith(("app/Solution.lean", "app/solution.dfy")):
                return m.name.rsplit("/", 1)[-1], t.extractfile(m).read().decode()
    return None, None


def plain(name, text):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        p.write_text(text)
        if name.endswith(".lean"):
            r = subprocess.run([LEAN, str(p)], capture_output=True, text=True, timeout=600)
            out = r.stdout + r.stderr
            return "ACCEPT" if r.returncode == 0 and ": error" not in out else "REJECT", ("sorry" in out)
        r = subprocess.run([DAFNY, "verify", "--allow-warnings", str(p)], capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
        return ("ACCEPT" if r.returncode == 0 and re.search(r"\b0 errors\b", out) else "REJECT"), False


for job in sys.argv[1:]:
    rows = []
    for rj in sorted(Path(job).glob("*/result.json")):
        tr = rj.parent
        ev = [json.loads(x) for x in (tr / "agent" / "fv.jsonl").read_text().splitlines()] if (tr / "agent" / "fv.jsonl").exists() else []
        ch = [e for e in ev if e.get("ev") == "check" and e.get("branch") == "B"]
        if not ch or ch[0]["status"] in ("PROVED", "CHECK_ERROR"):
            continue
        name, text = s0_solution(tr)
        if not text:
            continue
        verdict, sorry_warn = plain(name, text)
        rows.append({"trial": tr.name, "millennium_A": ch[0]["status"], "violations": ch[0].get("violations"),
                     "plain_A": verdict, "sorry_warning": sorry_warn, "millennium_B_final": ch[-1]["status"]})
    out = Path("runs/fv") / f"plain_{Path(job).name}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    acc = [r for r in rows if r["plain_A"] == "ACCEPT"]
    print(f"== {job}: first stops rejected by Millennium Harness: {len(rows)}; "
          f"accepted by the plain toolchain: {len(acc)}")
    for r in acc:
        print("   ", r["trial"], r["millennium_A"], "->", r["millennium_B_final"], "|", json.dumps(r["violations"])[:160])
