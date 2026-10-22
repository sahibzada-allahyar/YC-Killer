"""DEV-only quick look at FV probe trials from agent/fv.jsonl summaries (in-sandbox rung statuses; the fresh regrade is
authoritative). Usage: python bench/fv/quicklook.py runs/jobs/<job> [...]"""
import collections
import json
import sys
from pathlib import Path


def rows(job):
    for rj in sorted(Path(job).glob("*/result.json")):
        d = rj.parent
        r = json.loads(rj.read_text())
        ex = (r.get("exception_info") or {}).get("exception_type")
        s, checks = None, []
        f = d / "agent" / "fv.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                j = json.loads(line)
                if j.get("ev") == "summary":
                    s = j
                if j.get("ev") == "check":
                    checks.append((j.get("branch"), j.get("round"), j.get("status")))
        yield r["task_name"], ex, s, checks


for job in sys.argv[1:]:
    c, n = collections.Counter(), 0
    print("==", job)
    for task, ex, s, checks in rows(job):
        n += 1
        if not s:
            c["no-summary:" + str(ex)] += 1
            continue
        b = s.get("B") or {}
        first = (b.get("checks") or [None])[0]
        final = b.get("final")
        stop = (s.get("A") or {}).get("stop")
        cell = task[:2]
        c[(cell, "A=" + str(first), "B=" + str(final), "stop=" + str(stop))] += 1
    for k, v in sorted(c.items(), key=lambda x: str(x[0])):
        print(f"{v:4d}  {k}")
    print("trials", n)
