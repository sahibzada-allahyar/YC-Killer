"""Build sealed Millennium Harness suites for a Harbor dataset (specs via Batch API, suites built on Modal).

  python bench/suites.py submit  <harbor_dataset_dir> <name>
  python bench/suites.py collect <name>        # -> runs/suites/<name>/<task>.json  (inputs + agreed outputs only)

Specs see only the statement + public examples (from the task's config.json / instruction); never tests.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
K = 3

app = modal.App("millennium-harness-suites")
# Modal volumes and secrets keep their pre-rename (sorryless-*) names: they hold existing data.
vol = modal.Volume.from_name("sorryless-lcb", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11").pip_install("hypothesis==6.168.5")
         .add_local_python_source("millennium_harness"))


def load_problem(task_dir: Path):
    sys.path.insert(0, str(ROOT))
    from millennium_harness.problem import from_lcb_record
    cfg = task_dir / "tests" / "spec_view.json"   # exploratory no-examples variant: what the spec writer may see
    if not cfg.exists():
        cfg = task_dir / "tests" / "config.json"
    rec = json.loads(cfg.read_text())
    p, _ = from_lcb_record(rec, with_private=False)
    return p


@app.function(image=image, cpu=2, memory=4096, timeout=1800, max_containers=64)
def build_remote(task: str, problem: dict, specs: list) -> dict:
    import tempfile
    from millennium_harness.problem import Problem
    from millennium_harness.spec import Suite
    p = Problem(**problem)
    d = tempfile.mkdtemp()
    paths = []
    for i, code in enumerate(specs):
        f = f"{d}/spec_{i}.py"
        Path(f).write_text(code or "APPLICABLE = False\n")
        paths.append(f)
    s = Suite(p, paths, min_refs=2)
    out = f"{d}/suite.json"
    s.save(out)
    return {"task": task, "suite": json.loads(Path(out).read_text())}


def submit(dataset_dir: str, name: str):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    sys.path.insert(0, str(ROOT))
    from millennium_harness.spec import spec_request
    reqs, probs = [], {}
    for td in sorted(Path(dataset_dir).iterdir()):
        if not (td / "tests" / "config.json").exists():
            continue
        p = load_problem(td)
        probs[td.name] = p.__dict__
        for i in range(K):
            reqs.append(Request(custom_id=f"spec__{td.name}__{i}",
                                params=MessageCreateParamsNonStreaming(**spec_request(p))))
    out = ROOT / "runs" / "specs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "problems.json").write_text(json.dumps(probs))
    b = anthropic.Anthropic().messages.batches.create(requests=reqs)
    (out / "batch.json").write_text(json.dumps({"id": b.id, "n": len(reqs)}))
    print("batch", b.id, len(reqs), "requests")


def collect(name: str):
    import anthropic
    sys.path.insert(0, str(ROOT))
    from millennium_harness.llm import cost, text_of
    from millennium_harness.spec import extract_code
    out = ROOT / "runs" / "specs" / name
    bid = json.loads((out / "batch.json").read_text())["id"]
    c = anthropic.Anthropic()
    while (b := c.messages.batches.retrieve(bid)).processing_status != "ended":
        print(time.strftime("%H:%M:%S"), b.request_counts, flush=True)
        time.sleep(60)
    probs = json.loads((out / "problems.json").read_text())
    specs, usd = {t: [None] * K for t in probs}, 0.0
    for r in c.messages.batches.results(bid):
        _, task, i = r.custom_id.split("__")
        if r.result.type == "succeeded":
            usd += cost(r.result.message.usage.model_dump(), batch=True)
            specs[task][int(i)] = extract_code(text_of(r.result.message.content))
    (out / "specs.json").write_text(json.dumps(specs))
    with open(ROOT / "runs" / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "item": f"specs-{name}", "usd": round(usd, 4)}) + "\n")
    sd = ROOT / "runs" / "suites" / name
    sd.mkdir(parents=True, exist_ok=True)
    args = [(t, probs[t], specs[t]) for t in probs]
    ok = 0
    with app.run():
        for res in build_remote.starmap(args, return_exceptions=True, order_outputs=False):
            if isinstance(res, Exception):
                print("ERR", repr(res)[:200]); continue
            (sd / f"{res['task']}.json").write_text(json.dumps(res["suite"]))
            ok += res["suite"]["status"] == "OK"
    print(f"spec cost ${usd:.3f}; suites OK {ok}/{len(args)}")


if __name__ == "__main__":
    {"submit": lambda: submit(sys.argv[2], sys.argv[3]), "collect": lambda: collect(sys.argv[2])}[sys.argv[1]]()
