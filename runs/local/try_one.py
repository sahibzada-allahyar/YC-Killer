import json, sys, os, time
sys.path.insert(0, os.getcwd())
from sorryless.problem import from_lcb_record, same_output
from sorryless.spec import spec_request, extract_code, Suite
from sorryless.llm import anthropic_parallel, cost
from sorryless import run as runner
tid = sys.argv[1]
rec = json.load(open(f"runs/tasks/livecodebench/{tid}/tests/config.json"))
p, hidden = from_lcb_record(rec, with_private=True)
print(p.pid, p.kind, p.meta, "examples", len(p.examples), "hidden", len(hidden))
d = f"runs/local/{tid}"; os.makedirs(d, exist_ok=True)
t0 = time.time()
reqs = [spec_request(p) for _ in range(3)]
cand_req = {"model": "claude-haiku-5-5", "max_tokens": 16000, "output_config": {"effort": "medium"},
    "messages": [{"role": "user", "content": f"Solve in Python 3.\n{p.statement}\n\n{p.format_hint()}\n\nReturn only one ```python block."}]}
res = anthropic_parallel(reqs + [cand_req], workers=4)
tot = sum(cost(u) for _, u in res); print(f"LLM {time.time()-t0:.1f}s cost ${tot:.4f}", [u.get('output_tokens') for _, u in res])
paths = []
for i, (txt, _) in enumerate(res[:3]):
    code = extract_code(txt) or ""; pp = f"{d}/spec_{i}.py"; open(pp, "w").write(code); paths.append(pp)
cand = extract_code(res[3][0]) or ""; open(f"{d}/cand.py","w").write(cand)
t0 = time.time(); s = Suite(p, paths); print(f"suite {time.time()-t0:.1f}s", s.status, json.dumps(s.report)[:600])
r = s.check_code(cand, f"{d}/work"); print("check:", r.status, r.n_checked, r.n_disagree, (r.smallest.input[:80], r.smallest.expected[:40], (r.smallest.got or '')[:40], r.smallest.err) if r.smallest else None)
path = runner.prepare(cand, f"{d}/work")
hid = runner.run(p.kind, path, [x for x, _ in hidden], p.func_name, timeout=10)
ok = all(same_output(p.kind, e, o) for (_, e), (o, _) in zip(hidden, hid)); print("hidden pass:", ok, sum(same_output(p.kind, e, o) for (_, e), (o, _) in zip(hidden, hid)), "/", len(hidden))
