import json, sys, os
sys.path.insert(0, os.getcwd())
from sorryless.problem import from_lcb_record, same_output
from sorryless.spec import Suite
from sorryless import run as runner
tid = sys.argv[1]; d = f"runs/local/{tid}"
rec = json.load(open(f"runs/tasks/livecodebench/{tid}/tests/config.json"))
p, hidden = from_lcb_record(rec, with_private=True)
s = Suite(p, [f"{d}/spec_{i}.py" for i in range(3)])
for name in sys.argv[2:]:
    code = open(f"{d}/{name}").read()
    r = s.check_code(code, f"{d}/work")
    path = runner.prepare(code, f"{d}/work")
    hid = runner.run(p.kind, path, [x for x, _ in hidden], p.func_name, timeout=10)
    nh = sum(same_output(p.kind, e, o) for (_, e), (o, _) in zip(hidden, hid))
    sm = r.smallest
    print(name, "| sorryless:", r.status, f"{r.n_disagree}/{r.n_checked}", "| hidden:", f"{nh}/{len(hidden)}", "|", (repr(sm.input[:60]), sm.expected[:30], (sm.got or '')[:30], (sm.err or '')[:80]) if sm else "")
