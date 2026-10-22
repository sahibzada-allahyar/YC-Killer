"""E4a re-derivation gate: k=3 fresh-context derivations per problem (problem text only, never the agent's work),
pairwise equivalence via a cross-family judge, majority answer stored per task. Used by bench.agents:MathGated.

  python bench/math_gate.py submit            # Batch API: 3 independent Haiku derivations per MathArena task
  python bench/math_gate.py collect           # -> runs/matharena/rederive.json {task: {"majority": str|None, ...}}
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "matharena"
K = 3
PROMPT = ("Solve the following problem independently and carefully. Derive the result step by step, check limiting "
          "cases and units where relevant, and end with a single line of the form\nFINAL: <answer as one LaTeX "
          "expression>\n\n{problem}")
EQUIV = ("Are these two final answers to the same {cat} problem mathematically equivalent (same quantity; notation "
         "and algebraic rearrangement do not matter)?\nProblem:\n{problem}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\n"
         "Reply with exactly one line: VERDICT: EQUIVALENT or VERDICT: NOT_EQUIVALENT")


def equivalent(cat, problem, a, b, model="google/gemini-3.6-flash") -> bool:
    import requests
    if not a or not b:
        return False
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=300, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, json={
        "model": model, "max_tokens": 4000, "provider": {"data_collection": "deny"},
        "messages": [{"role": "user", "content": EQUIV.format(cat=cat, problem=problem, a=a, b=b)}]}).json()
    return "VERDICT: EQUIVALENT" in ((r.get("choices") or [{}])[0].get("message", {}).get("content", ""))


def submit():
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    gold = json.loads((OUT / "gold.json").read_text())
    reqs = [Request(custom_id=f"d__{t}__{i}", params=MessageCreateParamsNonStreaming(
        model="claude-haiku-5-5", max_tokens=32000, output_config={"effort": "medium"},
        messages=[{"role": "user", "content": PROMPT.format(problem=g["problem"])}]))
        for t, g in gold.items() for i in range(K)]
    b = anthropic.Anthropic().messages.batches.create(requests=reqs)
    (OUT / "rederive_batch.json").write_text(json.dumps({"id": b.id, "n": len(reqs)}))
    print("batch", b.id, len(reqs))


def collect():
    import anthropic
    from millennium_harness.llm import cost, text_of
    gold = json.loads((OUT / "gold.json").read_text())
    bid = json.loads((OUT / "rederive_batch.json").read_text())["id"]
    c = anthropic.Anthropic()
    while (b := c.messages.batches.retrieve(bid)).processing_status != "ended":
        print(time.strftime("%H:%M:%S"), b.request_counts, flush=True)
        time.sleep(60)
    ans, usd = {t: [None] * K for t in gold}, 0.0
    for r in c.messages.batches.results(bid):
        _, t, i = r.custom_id.split("__")
        if r.result.type == "succeeded":
            usd += cost(r.result.message.usage.model_dump(), batch=True)
            m = re.findall(r"FINAL:\s*(.+)", text_of(r.result.message.content))
            ans[t][int(i)] = m[-1].strip() if m else None

    def majority(t):
        a, g = ans[t], gold[t]
        for i in range(K):
            for j in range(i + 1, K):
                if a[i] and a[j] and equivalent(g["cat"], g["problem"], a[i], a[j]):
                    return {"majority": a[i], "answers": a}
        return {"majority": None, "answers": a}
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip(gold, ex.map(majority, gold)))
    (OUT / "rederive.json").write_text(json.dumps(res))
    with open(ROOT / "runs" / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "item": "e4a-rederive", "usd": round(usd, 4)}) + "\n")
    print(f"majority found for {sum(r['majority'] is not None for r in res.values())}/{len(res)}; cost ${usd:.3f}")


if __name__ == "__main__":
    {"submit": submit, "collect": collect}[sys.argv[1]]()
