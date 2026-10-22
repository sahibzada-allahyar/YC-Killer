"""Deterministic text audit of vericoding's published 'vericoded' (LLM-verified) solutions (no compiling, no models).

For every solution file: split by vericoding's own <vc-*> region markers, compare the frozen regions with the
benchmark record (runs/dafny/vericoding.jsonl), and scan the model-written regions (comments and strings removed) for
escape hatches. This is a lower bound on unsound 'verified' solutions; the compiled rung audit is separate.
  python bench/audit/vericoded_text.py VERICODED_DIR [OUT.jsonl]
"""
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HATCH = {
    "lean": [("sorry", r"\bsorry\b"), ("sorryAx", r"\bsorryAx\b"), ("admit", r"\badmit\b"), ("axiom", r"(?m)^\s*(?:private\s+|protected\s+)?axiom\b"),
             ("native_decide", r"\bnative_decide\b|decide\s*\+\s*native|Lean\.ofReduceBool"),
             ("implemented_by/extern", r"implemented_by|@\[\s*extern"), ("unsafe", r"\bunsafe\b"),
             ("debug/compiler option", r"set_option\s+(?:debug|compiler)\.")],
    "dafny": [("assume", r"\bassume\b"), ("{:axiom}", r"\{\s*:\s*axiom\b"), ("{:verify false}", r"\{\s*:\s*verify\s+false"),
              ("{:extern}", r"\{\s*:\s*extern\b"), ("decreases *", r"decreases\s*\*")],
    "verus": [("assume(", r"\bassume\s*\("), ("admit(", r"\badmit\s*\("), ("external_body", r"external_body"),
              ("external", r"verifier\s*(?:::|\()\s*external\b"), ("assume_specification", r"assume_specification")],
}
AGENT = {"lean": ["vc-helpers", "vc-definitions", "vc-theorems"], "dafny": ["vc-helpers", "vc-code"],
         "verus": ["vc-helpers", "vc-code"]}


def strip(code, lang):
    if lang == "lean":
        code = re.sub(r"/-.*?-/", " ", code, flags=re.S)
        code = re.sub(r"--[^\n]*", " ", code)
    else:
        code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
        code = re.sub(r"//[^\n]*", " ", code)
    return re.sub(r'"(?:\\.|[^"\\\n])*"', '""', code)


def regions(text, lang):
    c = "--" if lang == "lean" else "//"
    out = {}
    for m in re.finditer(rf"{re.escape(c)}\s*<(vc-[\w-]+)>\s*\n(.*?){re.escape(c)}\s*</\1>", text, re.S):
        out[m.group(1)] = out.get(m.group(1), "") + m.group(2)
    if "vc-preamble" not in out:  # vericoded Dafny files put the preamble before <vc-helpers> without a marker
        m = re.search(rf"{re.escape(c)}\s*<vc-", text)
        out["vc-preamble"] = text[:m.start()] if m else ""
    return out


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def headers_lean(block):
    """Frozen theorem/def statements = text up to ':=' of each declaration in the benchmark's sorry skeleton."""
    return [norm(h) for h in re.findall(r"((?:theorem|lemma|def|abbrev)\b.*?):=", block or "", re.S)]


def audit(path, rec, lang):
    text = path.read_text(errors="replace")
    reg = regions(text, lang)
    flags = []
    for name in AGENT[lang]:
        code = strip(reg.get(name, ""), lang)
        for label, pat in HATCH[lang]:
            if re.search(pat, code):
                flags.append(f"{label} in {name}")
    if not reg:
        flags.append("no vc markers")
    tamper = []
    if lang == "lean":
        if norm(strip(reg.get("vc-preamble") or "", lang)) != norm(strip(rec.get("vc-preamble") or "", lang)):
            tamper.append("vc-preamble differs")
        for blk in ("vc-definitions", "vc-theorems"):
            got = norm(reg.get(blk))
            for h in headers_lean(rec.get(blk)):
                if h and h not in got:
                    tamper.append(f"{blk} statement changed: {h[:60]}")
    else:
        for blk in ("vc-preamble", "vc-spec"):
            if blk in reg or rec.get(blk):
                if norm(strip(reg.get(blk) or "", lang)) != norm(strip(rec.get(blk) or "", lang)):
                    tamper.append(f"{blk} differs")
    return flags, tamper


def main(vdir, out=None):
    rows = {r["id"]: r for r in map(json.loads, open(ROOT / "runs/dafny/vericoding.jsonl"))}
    res = []
    for p in sorted(Path(vdir).rglob("*")):
        if p.suffix not in (".lean", ".dfy", ".rs"):
            continue
        m = re.match(r"([A-Z]{2}\d{4})", p.name)
        if not m or m.group(1) not in rows:
            continue
        rec = rows[m.group(1)]
        lang = rec["language"]
        model = p.parent.name if lang == "lean" else None
        flags, tamper = audit(p, rec, lang)
        res.append({"id": rec["id"], "lang": lang, "source": rec["source"], "model": model,
                    "clean_task": str(rec.get("qa-issue")) == "0", "file": str(p.relative_to(vdir)),
                    "hatches": flags, "tamper": tamper})
    if out:
        Path(out).write_text("".join(json.dumps(r) + "\n" for r in res))
    for clean in (True, False):
        sub = [r for r in res if r["clean_task"] == clean]
        print(f"\n== {'clean tasks (qa-issue==0)' if clean else 'tasks with a benchmark QA issue'}: {len(sub)} solutions")
        for lang in ("lean", "dafny", "verus"):
            s = [r for r in sub if r["lang"] == lang]
            h = [r for r in s if r["hatches"]]
            t = [r for r in s if r["tamper"]]
            either = [r for r in s if r["hatches"] or r["tamper"]]
            print(f"  {lang:6s} n={len(s):5d}  escape hatch in model code: {len(h):4d}  frozen text changed: {len(t):4d}"
                  f"  either: {len(either):4d} ({100 * len(either) / max(1, len(s)):.1f}%)")
            c = collections.Counter(f.split(" in ")[0] for r in h for f in r["hatches"])
            if c:
                print("         hatches:", dict(c.most_common()))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
