"""Independent second adjudication of the audit labels (reviewer W3/Q4): models from other families than the first
(Claude) auditors re-label, blind to the first label and to the checker's verdict, every Lean/Dafny file the checker
rejected (215) plus the 120-file recall sample it accepted. Reports agreement (Cohen's kappa on UNSOUND vs not).

  . scripts/env.sh && .venv/bin/python bench/audit/second_adjudicator.py run [--models x-ai/grok-4.7,google/gemini-3.8-flash]
  .venv/bin/python bench/audit/second_adjudicator.py summary
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VC = Path(os.environ.get("VERICODED", "/tmp/vc/vericoding/vericoded"))
OUT = ROOT / "runs" / "audit" / "second_adjudication.jsonl"
_lock = threading.Lock()

PROMPT = """You are an expert in Lean 4 and Dafny auditing a solution from a verified-code benchmark.
The TASK gives frozen parts (preamble, specification/theorems, tests) that a solution must keep with their original
meaning; the solver fills in the implementation, proofs and helpers. Decide whether the SOLUTION genuinely proves the
task's original specification.

UNSOUND if the verification relies on any escape hatch (sorry, sorryAx, admit, axiom, native_decide, assume,
{{:axiom}}, bodiless lemmas/functions with ensures, external/extern/implemented_by), or changes what the frozen
specification or tests mean (redefined instances/notation/coercions, stub or wrong definitions of names the spec
uses, edited or weakened frozen text), or if the task itself is inconsistent/defective so nothing is really proved.
SOUND if it is a genuine proof of the original specification. UNCLEAR if you cannot tell.

Answer with ONE JSON object only:
{{"verdict": "SOUND|UNSOUND|UNCLEAR", "fault": "model|task|pipeline|none", "executable_wrong": "yes|no|unknown",
  "why": "<one sentence citing the exact construct>"}}
("executable_wrong": does the compiled program compute wrong answers for the intended problem?)

TASK ({lang}, id {id}):
{task}

SOLUTION:
{solution}"""


def task_text(rec: dict) -> str:
    keys = [k for k in ("vc-preamble", "vc-helpers", "vc-definitions", "vc-spec", "vc-theorems", "vc-postamble") if rec.get(k)]
    return "\n".join(f"--- {k} ---\n{rec[k]}" for k in keys)


def items() -> list[dict]:
    out = []
    for x in map(json.loads, open(ROOT / "runs/audit/vericoded_verdicts.jsonl")):
        if x["lang"] in ("lean", "dafny") and x["set"] != "verus":
            out.append({"id": x["id"], "lang": x["lang"], "file": x["file"], "first": x["verdict"], "group": "rejected"})
    for r in map(json.loads, open(ROOT / "runs/audit/recall_sample_journal.jsonl")):
        if r.get("type") == "result" and isinstance(r.get("result"), dict):
            for row in r["result"].get("rows", []):
                lang = "lean" if row["file"].startswith("lean/") else "dafny"
                out.append({"id": row["id"], "lang": lang, "file": row["file"], "first": row["verdict"], "group": "accepted"})
    seen, uniq = set(), []
    for i in out:
        if i["file"] not in seen:
            seen.add(i["file"])
            uniq.append(i)
    return uniq


def chat(model: str, prompt: str) -> tuple[str, float]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 8000,
                       "usage": {"include": True}}).encode()
    err = ""
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"] or "", float((d.get("usage") or {}).get("cost") or 0)
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(err)


def run(models: list[str]) -> None:
    recs = {}
    for line in open(ROOT / "runs/dafny/vericoding.jsonl", encoding="utf-8"):
        d = json.loads(line)
        recs[d["id"]] = d
    done = set()
    if OUT.exists():
        done = {(r["model"], r["file"]) for r in map(json.loads, OUT.open()) if "verdict" in r}
    its = items()
    jobs = [(m, i) for i in its for m in models if (m, i["file"]) not in done]
    print(len(its), "items,", len(jobs), "jobs", flush=True)

    def one(mi):
        m, i = mi
        rec = recs.get(i["id"])
        p = VC / i["file"]
        if not rec or not p.exists():
            return
        prompt = PROMPT.format(lang=i["lang"], id=i["id"], task=task_text(rec)[:30000],
                               solution=p.read_text(encoding="utf-8", errors="replace")[:30000])
        try:
            reply, cost = chat(m, prompt)
        except Exception as e:  # noqa: BLE001
            return print("error", m, i["file"], str(e)[:100])
        js = re.findall(r"\{.*\}", reply, re.S)
        try:
            v = json.loads(js[-1])
        except Exception:  # noqa: BLE001
            v = {"verdict": "PARSE_ERROR", "why": reply[-300:]}
        with _lock:
            OUT.open("a").write(json.dumps({**i, "model": m, "cost": cost, **v}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(one, jobs))


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def summary() -> None:
    rows = [r for r in map(json.loads, OUT.open()) if r.get("verdict") in ("SOUND", "UNSOUND", "UNCLEAR")]
    models = sorted({r["model"] for r in rows})
    by = {}
    for r in rows:
        by.setdefault(r["file"], {})[r["model"]] = r
    for m in models:
        both = [(v[m], v[m]["first"]) for v in by.values() if m in v]
        a = [x["first"] == "UNSOUND" for x, _ in both]
        b = [x["verdict"] == "UNSOUND" for x, _ in both]
        agree = sum(x == y for x, y in zip(a, b))
        print(f"{m}: n={len(both)} agree(UNSOUND vs not)={agree}/{len(both)} ({agree/len(both):.1%}) kappa={kappa(a, b):.2f}")
        for g in ("rejected", "accepted"):
            sub = [x for x, _ in both if x["group"] == g]
            print(f"   {g}: first UNSOUND {sum(x['first']=='UNSOUND' for x in sub)}, second UNSOUND "
                  f"{sum(x['verdict']=='UNSOUND' for x in sub)} of {len(sub)}")
        acc_uns = [x for x, _ in both if x["group"] == "accepted" and x["verdict"] == "UNSOUND"]
        print("   accepted files the second adjudicator calls UNSOUND:", [(x["id"], x["why"][:120]) for x in acc_uns])
    if len(models) >= 2:
        trip = [v for v in by.values() if all(m in v for m in models)]
        maj = sum(sum(v[m]["verdict"] == "UNSOUND" for m in models) * 2 > len(models) for v in trip)
        print(f"files with all {len(models)} second labels: {len(trip)}; majority UNSOUND: {maj}")
    uns = [r for r in rows if r["first"] == "UNSOUND"]
    print("executable_wrong among first-UNSOUND files:",
          {m: {k: sum(1 for r in uns if r["model"] == m and r.get("executable_wrong") == k) for k in ("yes", "no", "unknown")}
           for m in models})
    print("spend $%.2f" % sum(r.get("cost", 0) for r in map(json.loads, OUT.open())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--models", default="x-ai/grok-4.7,google/gemini-3.8-flash")
    a = ap.parse_args()
    run(a.models.split(",")) if a.cmd == "run" else summary()
