import json, re
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
VC = Path('/tmp/vc/vericoding/vericoded')
RECS = {(r['language'], r['id']): r for r in map(json.loads, open(REPO/'runs/dafny/vericoding.jsonl'))}
ROWS = [json.loads(l) for l in open(REPO/'runs/audit/vericoded_rung.jsonl')]
TEXT = [json.loads(l) for l in open(REPO/'runs/audit/vericoded_text.jsonl')]
TARGET = [r for r in ROWS if r['baseline_status']=='ACCEPT' and r['sorryless_status'] not in ('PROVED','NOT_SAMPLED')]
TEXTHITS = [r for r in TEXT if r['hatches'] or r['tamper']]
def regions(text, lang):
    c = "--" if lang == "lean" else "//"
    out = {}
    for m in re.finditer(rf"{re.escape(c)}\s*<(vc-[\w-]+)>\s*\n(.*?){re.escape(c)}\s*</\1>", text, re.S):
        out[m.group(1)] = out.get(m.group(1), "") + m.group(2)
    if "vc-preamble" not in out:
        m = re.search(rf"{re.escape(c)}\s*<vc-", text)
        out["vc-preamble"] = text[:m.start()] if m else ""
    return out
def sol(file): return (VC/file).read_text()
def rec(lang, id): return RECS[(lang, id)]
