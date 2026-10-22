"""E1: fixed-pool selector study (go/no-go for Millennium Harness). Batch API for generation, Modal for execution.

  python bench/e1.py prepare            # Modal: download LCB v6 -> volume; select N problems; write runs/e1/problems.jsonl
  python bench/e1.py submit             # Batch API: 8 candidates + 3 specs + 3 CodeT test sets per problem
  python bench/e1.py collect            # poll + save runs/e1/gen/*.json + upload to volume
  python bench/e1.py evaluate           # Modal map: public/hidden/selector signals per candidate -> runs/e1/eval.jsonl
  python bench/e1.py judge-submit|judge-collect
  python bench/e1.py analyze            # selectors + G1 gate

Hidden tests never leave Modal; no model ever sees them. Prompts contain only statements + public examples.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
TAG = os.environ.get("E1_TAG", "e1")          # e.g. e1-low: candidates at another effort, same problems/specs
BASE = os.environ.get("E1_BASE", "e1")        # where problems/specs/codet come from
EFFORT = os.environ.get("E1_EFFORT", "medium")
RUNS = ROOT / "runs" / TAG
SEED = 20261011
N_PROBLEMS = int(os.environ.get("E1_N", 200))
N_CAND, N_SPEC, N_CODET = 8, 3, 3
MODEL = "claude-haiku-5-5"
LCB_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]

app = modal.App("millennium-harness-e1")
# Modal volumes and secrets keep their pre-rename (sorryless-*) names: they hold existing data.
vol = modal.Volume.from_name("sorryless-lcb", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("hypothesis==6.168.5", "huggingface_hub")
         .add_local_python_source("millennium_harness"))
V = "/vol"


# ---------------------------------------------------------------- Modal side

@app.function(image=image, volumes={V: vol}, cpu=2, memory=12288, timeout=3600)
def prepare_remote(exclude: list[str], n: int, seed: int) -> list[dict]:
    from huggingface_hub import hf_hub_download
    os.makedirs(f"{V}/raw", exist_ok=True)
    index = []  # (file, offset, pid, difficulty)
    for fn in LCB_FILES:
        path = hf_hub_download("livecodebench/code_generation_lite", fn, repo_type="dataset", local_dir=f"{V}/raw")
        with open(path, "rb") as f:
            off = 0
            for line in f:
                rec = json.loads(line)
                index.append((path, off, str(rec["question_id"]), rec.get("difficulty")))
                off += len(line)
    vol.commit()
    ex = {e.lower() for e in exclude}
    pool = sorted({(p, o, q, d) for p, o, q, d in index if d in ("medium", "hard") and q.lower() not in ex},
                  key=lambda t: t[2])
    rng = random.Random(seed)
    picked = rng.sample(pool, min(n, len(pool)))
    os.makedirs(f"{V}/e1/problems", exist_ok=True)
    public = []
    for path, off, pid, _ in picked:
        with open(path, "rb") as f:
            f.seek(off)
            rec = json.loads(f.readline())
        with open(f"{V}/e1/problems/{pid}.json", "w") as g:
            json.dump(rec, g)
        public.append({k: rec[k] for k in ("question_id", "question_title", "question_content", "platform",
                                           "contest_date", "starter_code", "difficulty", "public_test_cases",
                                           "metadata")})
    vol.commit()
    print(f"index={len(index)} pool={len(pool)} picked={len(picked)}")
    return public


def _codet_tests(texts: list[str]) -> list[tuple[str, str]]:
    import re
    tests = []
    for t in texts:
        for blk in re.findall(r"```(?:json)?\s*\n(.*?)```", t or "", re.S) or [t or ""]:
            try:
                arr = json.loads(blk)
            except Exception:
                continue
            for it in arr if isinstance(arr, list) else []:
                if isinstance(it, dict) and isinstance(it.get("input"), str) and isinstance(it.get("output"), str):
                    tests.append((it["input"], it["output"]))
    seen, out = set(), []
    for x in tests:
        if x[0] not in seen:
            seen.add(x[0]); out.append(x)
    return out[:30]


@app.function(image=image, volumes={V: vol}, cpu=2, memory=4096, timeout=3000, max_containers=64)
def evaluate_remote(pid: str, tag: str = "e1", base: str = "e1") -> dict:
    import tempfile
    from millennium_harness.problem import from_lcb_record, same_output
    from millennium_harness.spec import Suite
    from millennium_harness import run as runner

    rec = json.load(open(f"{V}/{base}/problems/{pid}.json"))
    gen = json.load(open(f"{V}/{tag}/gen/{pid}.json"))
    p, hidden = from_lcb_record(rec, with_private=True)
    work = tempfile.mkdtemp()
    t0 = time.time()
    out = {"pid": pid, "kind": p.kind, "difficulty": p.meta.get("difficulty"), "n_hidden": len(hidden),
           "cands": []}

    # public + hidden correctness (hidden: 30s/test like the LCB grader, fail-fast, 300s cap)
    for ci, code in enumerate(gen["cands"]):
        c = {"i": ci, "extracted": code is not None}
        if code is None:
            c.update(pub=False, hidden=False); out["cands"].append(c); continue
        path = runner.prepare(code, f"{work}/c{ci}")
        pr = runner.run(p.kind, path, [x for x, _ in p.examples], p.func_name, timeout=30)
        c["pub"] = all(same_output(p.kind, e, o) for (_, e), (o, _) in zip(p.examples, pr))
        ok, start = True, time.time()
        for k in range(0, len(hidden), 8):
            chunk = hidden[k:k + 8]
            hr = runner.run(p.kind, path, [x for x, _ in chunk], p.func_name, timeout=30)
            if not all(same_output(p.kind, e, o) for (_, e), (o, _) in zip(chunk, hr)):
                ok = False; break
            if time.time() - start > 300:
                ok = False; c["hidden_cap"] = True; break
        c["hidden"] = ok
        out["cands"].append(c)

    # spec suites
    sp = []
    for si, code in enumerate(gen["specs"]):
        f = f"{work}/spec_{si}.py"
        Path(f).write_text(code or "APPLICABLE = False\n")
        sp.append(f)
    suites = {"s4": Suite(p, sp, min_refs=2), "s4_k1": Suite(p, sp[:1], min_refs=1),
              "s4_nofilter": Suite(p, sp[:1], min_refs=1, filter=False)}
    out["suites"] = {k: {"status": s.status, "n_inputs": len(s.inputs), "report": s.report, "sha": s.sha}
                     for k, s in suites.items()}
    mbr_inputs = next((s for s in suites.values() if s.status == "OK"), None)
    for c in out["cands"]:
        code = gen["cands"][c["i"]]
        for k, s in suites.items():
            if code is None or s.status != "OK":
                c[k] = None; continue
            r = s.check_code(code, f"{work}/{k}_{c['i']}")
            c[k] = {"n_dis": r.n_disagree, "n": r.n_checked}
            if s is mbr_inputs:
                outs = r.outputs or []
                c["mbr_sig"] = __import__("hashlib").sha1(json.dumps(
                    [None if o is None else (" ".join(o.split()) if p.kind == "stdin" else o) for o in outs]
                ).encode()).hexdigest()[:16]

    # CodeT: candidates vs model-predicted tests
    ct = _codet_tests(gen.get("codet") or [])
    out["n_codet"] = len(ct)
    for c in out["cands"]:
        code = gen["cands"][c["i"]]
        if code is None or not ct:
            c["codet_pass"] = None; continue
        path = runner.prepare(code, f"{work}/ct{c['i']}")
        rr = runner.run(p.kind, path, [x for x, _ in ct], p.func_name, timeout=5)
        c["codet_pass"] = [bool(same_output(p.kind, e, o)) for (_, e), (o, _) in zip(ct, rr)]
    out["secs"] = round(time.time() - t0, 1)
    os.makedirs(f"{V}/{tag}/eval", exist_ok=True)
    Path(f"{V}/{tag}/eval/{pid}.json").write_text(json.dumps(out))
    vol.commit()
    return out


@app.function(image=image, volumes={V: vol}, cpu=2, memory=4096, timeout=3000, max_containers=64)
def diagnose_remote(pid: str, idxs: list[int]) -> dict:
    """Why do sample-passing candidates fail hidden tests? first failing test: timeout | error | wrong (+ input size)."""
    import tempfile
    from millennium_harness.problem import from_lcb_record, same_output
    from millennium_harness import run as runner
    rec = json.load(open(f"{V}/e1/problems/{pid}.json"))
    gen = json.load(open(f"{V}/e1/gen/{pid}.json"))
    p, hidden = from_lcb_record(rec, with_private=True)
    work, out = tempfile.mkdtemp(), {}
    for ci in idxs:
        path = runner.prepare(gen["cands"][ci], f"{work}/c{ci}")
        why = None
        for k in range(0, len(hidden), 8):
            chunk = hidden[k:k + 8]
            hr = runner.run(p.kind, path, [x for x, _ in chunk], p.func_name, timeout=30)
            for (x, e), (o, err) in zip(chunk, hr):
                if not same_output(p.kind, e, o):
                    why = {"reason": "timeout" if err == "timeout" else ("error" if err else "wrong"),
                           "input_len": len(x), "test_idx": k}
                    break
            if why:
                break
        out[ci] = why
    return {"pid": pid, "diag": out}


@app.function(image=image, volumes={V: vol, "/sealed": modal.Volume.from_name("sorryless-sealed")}, timeout=1800)
def copy_freshcp_remote(src: str, base: str) -> list[dict]:
    os.makedirs(f"{V}/{base}/problems", exist_ok=True)
    pub = []
    for t in sorted(os.listdir(f"/sealed/{src}/tasks")):
        rec = json.load(open(f"/sealed/{src}/tasks/{t}/tests/config.json"))
        with open(f"{V}/{base}/problems/{rec['question_id']}.json", "w") as g:
            json.dump(rec, g)
        pub.append({k: rec[k] for k in ("question_id", "question_title", "question_content", "platform",
                                         "contest_date", "starter_code", "difficulty", "public_test_cases", "metadata")})
    vol.commit()
    return pub


@app.function(image=image, volumes={V: vol}, cpu=2, memory=4096, timeout=3000, max_containers=64)
def respec_remote(pid: str, specs: list, cands: list, base_eval: dict, base: str = "e1") -> dict:
    """Rebuild suites from NEW specs and re-score the SAME candidates (hidden outcomes reused from base_eval)."""
    import tempfile
    from millennium_harness.problem import from_lcb_record
    from millennium_harness.spec import Suite
    rec = json.load(open(f"{V}/{base}/problems/{pid}.json"))
    p, _ = from_lcb_record(rec, with_private=False)
    work = tempfile.mkdtemp()
    sp = []
    for si, code in enumerate(specs):
        f = f"{work}/spec_{si}.py"
        Path(f).write_text(code or "APPLICABLE = False\n"); sp.append(f)
    suites = {"s4": Suite(p, sp, min_refs=2), "s4_k1": Suite(p, sp[:1], min_refs=1)}
    out = json.loads(json.dumps(base_eval))
    out["suites"] = {k: {"status": s.status, "n_inputs": len(s.inputs), "report": s.report} for k, s in suites.items()}
    for c in out["cands"]:
        code = cands[c["i"]]
        for k, s in suites.items():
            if code is None or s.status != "OK":
                c[k] = None; continue
            r = s.check_code(code, f"{work}/{k}_{c['i']}")
            c[k] = {"n_dis": r.n_disagree, "n": r.n_checked}
    return out


# ---------------------------------------------------------------- local side (prompts, batches, analysis)

def _problems():
    from millennium_harness.problem import from_lcb_record
    recs = [json.loads(l) for l in open(ROOT / "runs" / BASE / "problems.jsonl")]
    return [from_lcb_record(r)[0] for r in recs]


def cand_request(p) -> dict:
    if p.kind == "functional":
        fmt = (f"You will use the following starter code to write the solution to the problem and enclose your "
               f"code within delimiters.\n```python\n{p.starter_code}\n```")
    else:
        fmt = ("Read the inputs from stdin, solve the problem, and write the answer to stdout (do not directly "
               "test on the sample inputs). Enclose your code within delimiters as follows.\n```python\n"
               "# YOUR CODE HERE\n```")
    user = (f"### Question:\n{p.statement}\n\n### Format: {fmt}\n\n### Answer: (use the provided format with "
            "backticks)")
    return {"model": MODEL, "max_tokens": 16000, "output_config": {"effort": EFFORT},
            "system": "You are an expert Python programmer. You will be given a question (problem specification) "
                      "and will generate a correct Python program that matches the specification and passes all "
                      "tests.",
            "messages": [{"role": "user", "content": user}]}


def codet_request(p) -> dict:
    user = (f"Task statement:\n{p.statement}\n\n{p.format_hint()}\n\n{p.examples_block()}\n\n"
            "Write 10 NEW, diverse, small test cases for this task (edge cases included), each with the exact "
            "expected output, formatted exactly like the examples. Work the outputs out carefully by hand. "
            "Return a JSON list of objects {\"input\": str, \"output\": str} in one ```json block.")
    return {"model": MODEL, "max_tokens": 16000, "output_config": {"effort": "medium"},
            "messages": [{"role": "user", "content": user}]}


def _client():
    import anthropic
    return anthropic.Anthropic()


def submit():
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    from millennium_harness.spec import spec_request
    reqs = []
    for p in _problems():
        for i in range(N_CAND):
            reqs.append(Request(custom_id=f"cand__{p.pid}__{i}", params=MessageCreateParamsNonStreaming(**cand_request(p))))
        if TAG != BASE:
            continue
        for i in range(N_SPEC):
            reqs.append(Request(custom_id=f"spec__{p.pid}__{i}", params=MessageCreateParamsNonStreaming(**spec_request(p))))
        for i in range(N_CODET):
            reqs.append(Request(custom_id=f"codet__{p.pid}__{i}", params=MessageCreateParamsNonStreaming(**codet_request(p))))
    # cost guard: assume <=10K output tokens avg + 3K input, batch price
    est = len(reqs) * (3000 * 0.10 + 10000 * 0.50) / 1e6 * 0.5
    print(f"{len(reqs)} requests; worst-case-ish estimate ${est:.2f}")
    if est > float(os.environ.get("E1_MAX_USD", 12)):
        sys.exit("refusing: estimate above E1_MAX_USD")
    RUNS.mkdir(parents=True, exist_ok=True)
    b = _client().messages.batches.create(requests=reqs)
    (RUNS / "batch_gen.json").write_text(json.dumps({"id": b.id, "n": len(reqs), "t": time.time()}))
    print("batch", b.id, b.processing_status)


def _wait(batch_id):
    c = _client()
    while True:
        b = c.messages.batches.retrieve(batch_id)
        rc = b.request_counts
        print(time.strftime("%H:%M:%S"), b.processing_status, f"ok={rc.succeeded} err={rc.errored} proc={rc.processing}",
              flush=True)
        if b.processing_status == "ended":
            return c
        time.sleep(60)


def collect():
    from millennium_harness.llm import cost, text_of
    from millennium_harness.spec import extract_code
    bid = json.loads((RUNS / "batch_gen.json").read_text())["id"]
    c = _wait(bid)
    gen: dict = {}
    usd, errs = 0.0, 0
    for r in c.messages.batches.results(bid):
        kind, pid, i = r.custom_id.split("__")
        g = gen.setdefault(pid, {"cands": [None] * N_CAND, "specs": [None] * N_SPEC, "codet": [None] * N_CODET,
                                 "usage": {}})
        if r.result.type != "succeeded":
            errs += 1; continue
        m = r.result.message
        u = m.usage.model_dump()
        usd += cost(u, batch=True)
        g["usage"][f"{kind}{i}"] = {"in": u.get("input_tokens"), "out": u.get("output_tokens"),
                                    "stop": m.stop_reason}
        txt = text_of(m.content)
        if kind == "cand":
            g["cands"][int(i)] = extract_code(txt)
        elif kind == "spec":
            g["specs"][int(i)] = extract_code(txt)
        else:
            g["codet"][int(i)] = txt
    if TAG != BASE:
        for pid, g in gen.items():
            b = json.loads((ROOT / "runs" / BASE / "gen" / f"{pid}.json").read_text())
            g["specs"], g["codet"] = b["specs"], b["codet"]
    os.makedirs(RUNS / "gen", exist_ok=True)
    for pid, g in gen.items():
        (RUNS / "gen" / f"{pid}.json").write_text(json.dumps(g))
    with vol.batch_upload(force=True) as up:
        up.put_directory(str(RUNS / "gen"), f"/{TAG}/gen")
    _ledger(f"{TAG}-gen", usd)
    print(f"collected {len(gen)} problems, errors={errs}, cost=${usd:.3f}")


def evaluate():
    pids = [p.pid for p in _problems()]
    done = {json.loads(l)["pid"] for l in open(RUNS / "eval.jsonl")} if (RUNS / "eval.jsonl").exists() else set()
    todo = [x for x in pids if x not in done]
    print(f"evaluating {len(todo)} problems on Modal")
    with app.run(), open(RUNS / "eval.jsonl", "a") as f:
        for res in evaluate_remote.map(todo, [TAG] * len(todo), [BASE] * len(todo), return_exceptions=True,
                                       order_outputs=False):
            if isinstance(res, Exception):
                print("ERR", repr(res)[:200]); continue
            f.write(json.dumps(res) + "\n"); f.flush()
            print(res["pid"], res["secs"], {k: v["status"] for k, v in res["suites"].items()},
                  sum(c.get("hidden", False) for c in res["cands"]), flush=True)


def judge_submit():
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    ev = {json.loads(l)["pid"]: json.loads(l) for l in open(RUNS / "eval.jsonl")}
    reqs = []
    for p in _problems():
        e = ev.get(p.pid)
        if not e:
            continue
        g = json.loads((RUNS / "gen" / f"{p.pid}.json").read_text())
        idx = [c["i"] for c in e["cands"] if c.get("pub")] or [c["i"] for c in e["cands"] if c["extracted"]]
        if len(idx) < 2:
            continue
        body = "\n\n".join(f"### Candidate {j}\n```python\n{g['cands'][j]}\n```" for j in idx)
        user = (f"Task statement:\n{p.statement}\n\n{p.format_hint()}\n\n{body}\n\nWhich candidate is most likely "
                f"fully correct (including edge cases and efficiency)? Answer with exactly `ANSWER: <number>` from "
                f"{idx}.")
        reqs.append(Request(custom_id=f"judge__{p.pid}__0", params=MessageCreateParamsNonStreaming(
            model=MODEL, max_tokens=16000, output_config={"effort": "medium"},
            messages=[{"role": "user", "content": user}])))
    b = _client().messages.batches.create(requests=reqs)
    (RUNS / "batch_judge.json").write_text(json.dumps({"id": b.id, "n": len(reqs)}))
    print("judge batch", b.id, len(reqs))


def judge_collect():
    import re
    from millennium_harness.llm import cost, text_of
    bid = json.loads((RUNS / "batch_judge.json").read_text())["id"]
    c = _wait(bid)
    picks, usd = {}, 0.0
    for r in c.messages.batches.results(bid):
        _, pid, _ = r.custom_id.split("__")
        if r.result.type != "succeeded":
            continue
        usd += cost(r.result.message.usage.model_dump(), batch=True)
        m = re.findall(r"ANSWER:\s*(\d+)", text_of(r.result.message.content))
        picks[pid] = int(m[-1]) if m else None
    (RUNS / "judge.json").write_text(json.dumps(picks))
    _ledger("e1-judge", usd)
    print(f"judge picks {len(picks)} cost ${usd:.3f}")


def diagnose():
    ev = [json.loads(l) for l in open(RUNS / "eval.jsonl")]
    jobs = [(e["pid"], [c["i"] for c in e["cands"] if c.get("pub") and not c.get("hidden")]) for e in ev]
    jobs = [j for j in jobs if j[1]]
    print(f"diagnosing {sum(len(j[1]) for j in jobs)} pub-pass/hidden-fail candidates in {len(jobs)} problems")
    res = []
    with app.run():
        for r in diagnose_remote.starmap(jobs, return_exceptions=True, order_outputs=False):
            if not isinstance(r, Exception):
                res.append(r)
    (RUNS / "diagnose.json").write_text(json.dumps(res))
    from collections import Counter
    print(Counter((d or {}).get("reason") for r in res for d in r["diag"].values()))


def import_freshcp():
    """E1_BASE=<name> SRC=<VAL|TEST>: copy FreshCP task configs (sealed volume) into /<BASE>/problems; public fields local."""
    src = os.environ["SRC"]
    sealed = modal.Volume.from_name("sorryless-sealed")
    pub = []
    with app.run():
        pub = copy_freshcp_remote.remote(src, BASE)
    RUNS_B = ROOT / "runs" / BASE
    RUNS_B.mkdir(parents=True, exist_ok=True)
    with open(RUNS_B / "problems.jsonl", "w") as f:
        for r in pub:
            f.write(json.dumps(r) + "\n")
    print(len(pub), "problems imported into", BASE)


def respec_submit():
    """E1_TAG=<new tag>: new spec batch for the BASE problems (current prompt)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    from millennium_harness.spec import spec_request
    reqs = [Request(custom_id=f"spec__{p.pid}__{i}", params=MessageCreateParamsNonStreaming(**spec_request(p)))
            for p in _problems() for i in range(N_SPEC)]
    RUNS.mkdir(parents=True, exist_ok=True)
    b = _client().messages.batches.create(requests=reqs)
    (RUNS / "batch_spec.json").write_text(json.dumps({"id": b.id, "n": len(reqs)}))
    print("spec batch", b.id, len(reqs))


def _openrouter_spec(args):
    import requests
    from millennium_harness.spec import extract_code, spec_request
    p, model = args
    req = spec_request(p)
    for attempt in range(3):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=600, headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, json={
                "model": model, "max_tokens": 16000, "provider": {"data_collection": "deny"},
                "messages": [{"role": "system", "content": req["system"]}] + req["messages"]}).json()
            return extract_code(r["choices"][0]["message"]["content"]), float((r.get("usage") or {}).get("cost") or 0)
        except Exception:  # noqa: BLE001
            time.sleep(5)
    return None, 0.0


def respec_openrouter():
    """Exploratory: E1_TAG=<tag> SPEC_MODELS=m1,m2,m3 — one spec per model (cross-family references), re-score."""
    from concurrent.futures import ThreadPoolExecutor
    models = os.environ["SPEC_MODELS"].split(",")
    probs = _problems()
    jobs = [(p, m) for p in probs for m in models]
    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(_openrouter_spec, jobs))
    specs, usd = {}, 0.0
    for (p, m), (code, c) in zip(jobs, out):
        specs.setdefault(p.pid, []).append(code)
        usd += c
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "specs.json").write_text(json.dumps(specs))
    _ledger(f"{TAG}-openrouter-specs", usd)
    print(f"specs from {models}: cost ${usd:.3f}")
    _rescore(specs)


def _rescore(specs):
    base = {json.loads(l)["pid"]: json.loads(l) for l in open(ROOT / "runs" / BASE / "eval.jsonl")}
    args = []
    for pid, sp in specs.items():
        if pid in base:
            g = json.loads((ROOT / "runs" / BASE / "gen" / f"{pid}.json").read_text())
            args.append((pid, sp, g["cands"], base[pid], BASE))
    print(f"re-scoring {len(args)} problems")
    with app.run(), open(RUNS / "eval.jsonl", "w") as f:
        for res in respec_remote.starmap(args, return_exceptions=True, order_outputs=False):
            if isinstance(res, Exception):
                print("ERR", repr(res)[:200]); continue
            f.write(json.dumps(res) + "\n"); f.flush()


def respec_collect():
    from millennium_harness.llm import cost, text_of
    from millennium_harness.spec import extract_code
    bid = json.loads((RUNS / "batch_spec.json").read_text())["id"]
    c = _wait(bid)
    specs, usd = {}, 0.0
    for r in c.messages.batches.results(bid):
        _, pid, i = r.custom_id.split("__")
        specs.setdefault(pid, [None] * N_SPEC)
        if r.result.type == "succeeded":
            usd += cost(r.result.message.usage.model_dump(), batch=True)
            specs[pid][int(i)] = extract_code(text_of(r.result.message.content))
    _ledger(f"{TAG}-specs", usd)
    (RUNS / "specs.json").write_text(json.dumps(specs))
    base = {json.loads(l)["pid"]: json.loads(l) for l in open(ROOT / "runs" / BASE / "eval.jsonl")}
    args = []
    for pid, sp in specs.items():
        if pid in base:
            g = json.loads((ROOT / "runs" / BASE / "gen" / f"{pid}.json").read_text())
            args.append((pid, sp, g["cands"], base[pid], BASE))
    print(f"re-scoring {len(args)} problems; spec cost ${usd:.3f}")
    with app.run(), open(RUNS / "eval.jsonl", "w") as f:
        for res in respec_remote.starmap(args, return_exceptions=True, order_outputs=False):
            if isinstance(res, Exception):
                print("ERR", repr(res)[:200]); continue
            f.write(json.dumps(res) + "\n"); f.flush()


def _ledger(item: str, usd: float):
    p = ROOT / "runs" / "ledger.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "item": item, "usd": round(usd, 4)}) + "\n")


def prepare():
    exclude = sorted(os.listdir(ROOT / "runs" / "tasks" / "livecodebench"))
    with app.run():
        pub = prepare_remote.remote(exclude, N_PROBLEMS, SEED)
    os.makedirs(RUNS, exist_ok=True)
    with open(RUNS / "problems.jsonl", "w") as f:
        for r in pub:
            f.write(json.dumps(r) + "\n")
    kinds = {}
    for r in pub:
        kinds[r["platform"]] = kinds.get(r["platform"], 0) + 1
    print(len(pub), "problems", kinds)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    cmd = sys.argv[1]
    {"prepare": prepare, "submit": submit, "collect": collect, "evaluate": evaluate,
     "judge-submit": judge_submit, "judge-collect": judge_collect, "diagnose": diagnose,
     "respec-submit": respec_submit, "respec-collect": respec_collect,
     "import-freshcp": import_freshcp, "respec-openrouter": respec_openrouter}[cmd]()
