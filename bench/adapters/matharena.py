"""MathArena post-cutoff sets (ArXivMath / ArXivPhys 08-2026, CC-BY-SA-4.0) -> Harbor tasks + offline judge grading.

  python bench/adapters/matharena.py build OUT_DIR            # tasks; gold answers kept OUT of the task env
  python bench/adapters/matharena.py grade JOB_DIR [JOB_DIR..] # judge answers vs gold (cross-family judge)

The agent writes its final answer (LaTeX) to /app/answer.txt; the in-sandbox verifier only records that an answer
exists (reward is assigned offline by an equivalence judge with the gold answer, so gold never enters the sandbox).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = {"arxivmath-0826": "math", "arxivphys-0826": "phys"}
GOLD = ROOT / "runs" / "matharena" / "gold.json"

DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir sympy==1.14.0 numpy==2.3.3 scipy==1.16.2 mpmath==1.3.0
"""
INSTRUCTION = """=== PROBLEM ===

{problem}

=== YOUR TASK ===
Solve the problem. You may use Python (sympy, numpy, scipy, mpmath are installed) for derivations and numeric checks.
When you are done, write ONLY your final answer, as a single LaTeX expression, to the file /app/answer.txt.
"""
TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
if [ -s /app/answer.txt ]; then cp /app/answer.txt /logs/verifier/answer.txt; echo 0 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""
TASK_TOML = """version = "1.0"
[metadata]
author_name = "MathArena (CC-BY-SA-4.0) via Millennium Harness adapter"
category = "{cat}"
[verifier]
timeout_sec = 60.0
[agent]
timeout_sec = 900.0
[environment]
build_timeout_sec = 600.0
cpus = 1
memory = "2G"
storage = "4G"
"""

JUDGE = """You are grading a final answer against a reference answer for a research-level {cat} problem.
Decide whether the candidate answer is mathematically equivalent to the reference (same quantity; algebraic
rearrangements, notation changes and equivalent closed forms count as equivalent; missing factors, wrong signs,
wrong limits or extra terms do not). Problem:
{problem}

Reference answer:
{gold}

Candidate answer:
{cand}

Reply with exactly one line: VERDICT: EQUIVALENT or VERDICT: NOT_EQUIVALENT"""


def _load():
    import pandas as pd
    rows = []
    for ds, cat in SETS.items():
        url = f"https://huggingface.co/datasets/MathArena/{ds}/resolve/main/data/train-00000-of-00001.parquet"
        df = pd.read_parquet(url)
        for _, r in df.iterrows():
            rows.append({"id": f"{cat}_{int(r['problem_idx']):03d}", "cat": cat, "problem": r["problem"],
                         "answer": r["answer"], "source": r.get("source")})
    return rows


def build(out: str):
    rows = _load()
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    GOLD.write_text(json.dumps({r["id"]: {"answer": r["answer"], "problem": r["problem"], "cat": r["cat"]} for r in rows}))
    for r in rows:
        td = Path(out) / r["id"]
        (td / "environment").mkdir(parents=True, exist_ok=True)
        (td / "tests").mkdir(parents=True, exist_ok=True)
        (td / "solution").mkdir(parents=True, exist_ok=True)
        (td / "task.toml").write_text(TASK_TOML.format(cat=r["cat"]))
        (td / "instruction.md").write_text(INSTRUCTION.format(problem=r["problem"]))
        (td / "environment" / "Dockerfile").write_text(DOCKERFILE)
        (td / "tests" / "test.sh").write_text(TEST_SH)
        os.chmod(td / "tests" / "test.sh", 0o755)
        (td / "solution" / "solve.sh").write_text("echo '' > /app/answer.txt\n")
    print(f"built {len(rows)} tasks into {out} (gold stored separately in {GOLD})")


def _judge(cat, problem, gold, cand, model="google/gemini-3.6-flash"):
    import requests
    if not cand or not cand.strip():
        return False
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=300, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, json={
        "model": model, "max_tokens": 4000, "provider": {"data_collection": "deny"},
        "messages": [{"role": "user", "content": JUDGE.format(cat=cat, problem=problem, gold=gold, cand=cand)}]}).json()
    txt = (r.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return "VERDICT: EQUIVALENT" in txt


def grade(jobs: list[str]):
    from concurrent.futures import ThreadPoolExecutor
    gold = json.loads(GOLD.read_text())
    items = []
    for j in jobs:
        for rj in Path(j).glob("*/result.json"):
            d = json.loads(rj.read_text())
            ans = rj.parent / "verifier" / "answer.txt"
            items.append((j, d["task_name"], ans.read_text() if ans.exists() else ""))

    def one(it):
        j, t, a = it
        g = gold[t]
        return {"job": Path(j).name, "task": t, "cat": g["cat"], "ok": _judge(g["cat"], g["problem"], g["answer"], a)}
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, items))
    out = ROOT / "runs" / "matharena" / f"grades_{int(time.time())}.json"
    out.write_text(json.dumps(res))
    from collections import defaultdict
    agg = defaultdict(list)
    for r in res:
        agg[(r["job"], r["cat"])].append(r["ok"])
    for k, v in sorted(agg.items()):
        print(k, f"{100 * sum(v) / len(v):.1f}% ({sum(v)}/{len(v)})")


if __name__ == "__main__":
    {"build": lambda: build(sys.argv[2]), "grade": lambda: grade(sys.argv[2:])}[sys.argv[1]]()
