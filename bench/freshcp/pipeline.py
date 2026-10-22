"""FreshCP — a sealed, post-cutoff competitive-programming test set from AtCoder (built entirely on Modal).

  python bench/freshcp/pipeline.py scrape VAL|TEST      # statements + official editorial code -> volume
  python bench/freshcp/pipeline.py build  VAL|TEST      # refs, inputs, labels, strength -> Harbor tasks on volume
  python bench/freshcp/pipeline.py stats  VAL|TEST      # aggregate stats only (safe to view)
  python bench/freshcp/pipeline.py fetch  VAL|TEST DIR  # download Harbor task dirs (TEST -> sealed/, never read it)

Labels come from OFFICIAL editorial solutions (C++/Python) that reproduce every sample; inputs come from generators
written by a different model family (Gemini 3.6 Flash via OpenRouter, data collection denied) and are kept only where all
working references agree. Strength = fraction of sample-passing mutants of the reference that the tests kill.
Windows: VAL = contests 2026-04-01..06-30 (pre-cutoff, inspectable); TEST = 2026-07-01..freeze (sealed).
Scraping is polite (1 request / 2 s, identified UA); bot challenges are never bypassed.
"""
from __future__ import annotations

import base64
import html
import json
import os
import pickle
import re
import sys
import time
import zlib
from html.parser import HTMLParser
from pathlib import Path

import modal

_P = Path(__file__).resolve().parents
ROOT = _P[2] if len(_P) > 2 else Path("/root")
WINDOWS = {"VAL": ("2026-04-01", "2026-07-01"), "TEST": ("2026-07-01", os.environ.get("FRESHCP_FREEZE", "2026-10-18"))}
LETTERS = {"abc": "cdefg", "arc": "abcd", "agc": "ab"}
UA = "millennium-harness-research/0.1 (academic benchmark; polite 1 req/2s)"
GEN_MODEL = "google/gemini-3.6-flash"

app = modal.App("millennium-harness-freshcp")
# Modal volumes and secrets keep their pre-rename (sorryless-*) names: they hold existing data.
vol = modal.Volume.from_name("sorryless-sealed", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("g++", "git", "curl")
         .run_commands("git clone --depth 1 https://github.com/atcoder/ac-library /opt/ac-library")
         .pip_install("requests==2.32.3")
         .add_local_python_source("millennium_harness"))
V = "/sealed"


# ------------------------------------------------------------------ scraping

class _Text(HTMLParser):
    """Extract the English statement (span.lang-en) as LCB-style plain text, plus samples."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0  # >0 while inside span.lang-en
        self.buf: list[str] = []
        self.sections: list[tuple[str, str]] = []
        self.h3 = None
        self.in_pre = False
        self.cur: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.depth == 0:
            if tag == "span" and "lang-en" in (a.get("class") or ""):
                self.depth = 1
            return
        if tag == "span":
            self.depth += 1
        if tag == "h3":
            self.h3 = ""
        elif tag == "pre":
            self.in_pre = True
            self.cur = []
        elif tag == "li":
            self.buf.append("\n- ")
        elif tag in ("p", "div", "br", "section"):
            self.buf.append("\n")

    def handle_endtag(self, tag):
        if self.depth == 0:
            return
        if tag == "span":
            self.depth -= 1
            return
        if tag == "h3" and self.h3 is not None:
            self.buf.append(f"\n\n{self.h3.strip()}\n\n")
            self.last_h3, self.h3 = self.h3.strip(), None
        elif tag == "pre":
            txt = "".join(self.cur)
            self.in_pre = False
            self.buf.append(txt.strip("\n") + "\n")
            self.sections.append((getattr(self, "last_h3", ""), txt))
        elif tag in ("p", "li"):
            self.buf.append("\n")

    def handle_data(self, data):
        if self.depth == 0:
            return
        if self.h3 is not None:
            self.h3 += data
            return
        if self.in_pre:
            self.cur.append(data)
        self.buf.append(data)

    def result(self):
        text = re.sub(r"[ \t]+\n", "\n", "".join(self.buf))
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        ins = {re.sub(r"\D", "", h): t for h, t in self.sections if h.lower().startswith("sample input")}
        outs = {re.sub(r"\D", "", h): t for h, t in self.sections if h.lower().startswith("sample output")}
        samples = [(ins[k].lstrip("\n"), outs[k].lstrip("\n")) for k in sorted(ins, key=lambda s: int(s or 0)) if k in outs]
        return text, samples


def _get(sess, url):
    time.sleep(2.0)
    r = sess.get(url, headers={"User-Agent": UA}, timeout=30, allow_redirects=False)
    if r.status_code != 200:
        return None, r.status_code
    if "cf-challenge" in r.text or "captcha" in r.text.lower()[:5000]:
        return None, "challenge"
    return r.text, 200


@app.function(image=image, volumes={V: vol}, timeout=6 * 3600, cpu=1)
def scrape_remote(window: str, contests: list[dict]) -> dict:
    import requests
    sess = requests.Session()
    stats = {"contests": 0, "tasks": 0, "with_code": 0, "errors": []}
    root = Path(f"{V}/{window}/raw")
    root.mkdir(parents=True, exist_ok=True)
    for c in contests:
        cid = c["id"]
        page, st = _get(sess, f"https://atcoder.jp/contests/{cid}/tasks?lang=en")
        if not page:
            stats["errors"].append(f"{cid}: tasks {st}"); continue
        tasks = sorted(set(re.findall(rf'href="/contests/{cid}/tasks/({cid}_[a-z0-9]+)"', page)))
        ed, _ = _get(sess, f"https://atcoder.jp/contests/{cid}/editorial?lang=en")
        ed_links = {}
        for part in re.split(r"<h3>", ed or "")[1:]:
            title = html.unescape(re.sub("<[^>]+>", "", part.split("</h3>")[0])).strip()
            m = re.match(r"([A-Za-z]+)\s*-", title)
            if m:
                ed_links[m.group(1).lower()] = re.findall(rf'href="(/contests/{cid}/editorial/\d+)"', part)
        stats["contests"] += 1
        for t in tasks:
            letter = t.split("_")[-1]
            if letter not in LETTERS.get(cid[:3], ""):
                continue
            if (root / f"{t}.json").exists():
                continue
            sp, st = _get(sess, f"https://atcoder.jp/contests/{cid}/tasks/{t}?lang=en")
            if not sp:
                stats["errors"].append(f"{t}: {st}"); continue
            tl = re.search(r"Time Limit:\s*([\d.]+)\s*sec", sp)
            title = re.search(r"<title>(.*?)</title>", sp)
            codes = []
            for link in ed_links.get(letter, [])[:5]:
                ep, _ = _get(sess, f"https://atcoder.jp{link}?lang=en")
                for blk in re.findall(r"<pre[^>]*>(.*?)</pre>", ep or "", re.S):
                    code = html.unescape(re.sub(r"<[^>]+>", "", blk))
                    lang = ("cpp" if ("#include" in code or "int main" in code) else
                            "py" if re.search(r"\binput\(|sys\.stdin|print\(", code) else None)
                    if lang and len(code) > 40:
                        codes.append({"lang": lang, "code": code, "src": link})
            rec = {"task": t, "contest": cid, "letter": letter, "date": c["date"],
                   "title": html.unescape(title.group(1)) if title else t,
                   "time_limit": float(tl.group(1)) if tl else 2.0, "html": sp, "codes": codes}
            (root / f"{t}.json").write_text(json.dumps(rec))
            stats["tasks"] += 1
            stats["with_code"] += bool(codes)
        vol.commit()
    return stats


# ------------------------------------------------------------------ building

FILTERS = {
    "interactive": r"\binteractive\b|\bflush\b",
    "special_judge": r"multiple (solutions|answers|correct|possible)|any of them|print any|output any|"
                     r"may (print|output) any|any one of them|any such|any valid",
    "float": r"absolute or relative error|relative error|absolute error|error of at most|error is at most",
    "image": r"<img",
}

GEN_PROMPT = """You write TEST INPUT GENERATORS for a competitive programming problem. Output two Python 3 programs.

1) gen.py — usage: `python3 gen.py SEED MODE`, MODE in {small, medium, large, edge}. Print exactly ONE valid test
   input to stdout, in exactly the problem's input format. Use random.Random(int(SEED)) only.
   - small: tiny sizes and values (e.g. N <= 8), many different shapes
   - medium: moderate sizes (e.g. N around 100..2000)
   - large: close to the maximum constraints, but keep the whole input under 4 MB
   - edge: tricky cases: minimum sizes, all-equal values, maximum values, sorted / reverse-sorted, degenerate
     structures, and anything specific to this problem that a wrong solution might mishandle
   Every input MUST satisfy every constraint and guarantee in the statement (e.g. distinctness, connectivity,
   'an answer exists'). If a guarantee is hard to sample, construct inputs that satisfy it by design.
2) validate.py — reads one input from stdin; exit code 0 iff it satisfies ALL constraints and the format, else 1.

Reply with exactly two ```python blocks: first gen.py, then validate.py."""


def _decode_private(tests):
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()


def _sh(cmd, inp=None, timeout=10, cwd=None):
    import subprocess
    try:
        p = subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=timeout, cwd=cwd, shell=isinstance(cmd, str))
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -9, None, "timeout"


MUTATIONS = [(r"<=", "<"), (r"(?<![<>=!])<(?![<=])", "<="), (r">=", ">"), (r"(?<![<>=-])>(?![>=])", ">="),
             (r"\+ ?1\b", ""), (r"- ?1\b", ""), (r"&&", "||"), (r"\|\|", "&&"), (r"= 0;", "= 1;"),
             (r"\+\+", "--")]


def _strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", ln) for ln in code.splitlines())


def _mutants(code: str, n: int, rng):
    """Single-site mutants in executable code only (comments, preprocessor lines and string literals excluded)."""
    code = _strip_comments(code)
    masked = []  # positions that must not be mutated
    for m in re.finditer(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', code):
        masked.append((m.start(), m.end()))
    for m in re.finditer(r"^\s*#.*$", code, re.M):
        masked.append((m.start(), m.end()))
    ok = lambda m: not any(a <= m.start() < b for a, b in masked)
    out = []
    spots = [(i, m) for i, (pat, _) in enumerate(MUTATIONS) for m in re.finditer(pat, code) if ok(m)]
    rng.shuffle(spots)
    for i, m in spots[: n * 3]:
        rep = MUTATIONS[i][1]
        mc = code[: m.start()] + rep + code[m.end():]
        if mc != code:
            out.append(mc)
        if len(out) >= n:
            break
    return out


@app.function(image=image, volumes={V: vol}, timeout=3600, cpu=2, memory=4096, max_containers=24,
              secrets=[modal.Secret.from_name("sorryless-openrouter")])
def build_remote(window: str, task: str, tmpl: dict) -> dict:
    import random
    import requests
    from millennium_harness.problem import same_output

    raw = json.loads(Path(f"{V}/{window}/raw/{task}.json").read_text())
    st = {"task": task, "contest": raw["contest"], "letter": raw["letter"], "date": raw["date"]}
    p = _Text()
    p.feed(raw["html"])
    text, samples = p.result()
    i = raw["html"].find('class="lang-en"')
    stmt_html = raw["html"][i:i + 60000] if i >= 0 else ""
    j = stmt_html.find('class="lang-ja"')
    stmt_html = stmt_html[:j] if j > 0 else stmt_html
    for name, pat in FILTERS.items():
        if re.search(pat, (text if name != "image" else stmt_html), re.I):
            return {**st, "status": f"filtered:{name}"}
    if not samples:
        return {**st, "status": "no_samples"}
    w = Path(f"/tmp/{task}")
    w.mkdir(exist_ok=True)
    # references: official editorial code that reproduces every sample
    refs = []
    for i, c in enumerate({c["code"]: c for c in raw["codes"]}.values()):
        if c["lang"] == "cpp":
            (w / f"r{i}.cpp").write_text(c["code"])
            rc, _, err = _sh(["g++", "-std=gnu++20", "-O2", "-I/opt/ac-library", "-o", str(w / f"r{i}"), str(w / f"r{i}.cpp")], timeout=120)
            if rc != 0:
                continue
            cmd = [str(w / f"r{i}")]
        else:
            (w / f"r{i}.py").write_text(c["code"])
            cmd = [sys.executable, str(w / f"r{i}.py")]
        if all(same_output("stdin", so, _sh(cmd, si, timeout=10)[1]) for si, so in samples):
            refs.append({"cmd": cmd, "lang": c["lang"], "src": c["src"]})
    st["n_refs"] = len(refs)
    if not refs:
        return {**st, "status": "no_working_ref"}
    # input generators from a different model family
    gen_ok, inputs = False, [s for s, _ in samples]
    cache = Path(f"{V}/{window}/gen/{task}")
    if (cache / "gen.py").exists():
        (w / "gen.py").write_text((cache / "gen.py").read_text()); (w / "validate.py").write_text((cache / "validate.py").read_text())
    for attempt in range(2):
        if (cache / "gen.py").exists() and attempt == 0:
            got = []
            for mode, k in (("small", 16), ("medium", 10), ("large", 4), ("edge", 10)):
                for s in range(k):
                    rc, out, _ = _sh([sys.executable, str(w / "gen.py"), str(s), mode], timeout=20)
                    if rc == 0 and out and out.strip() and len(out) < 6_000_000:
                        if _sh([sys.executable, str(w / "validate.py")], out, timeout=30)[0] == 0:
                            got.append(out)
            if len(set(got)) >= 15:
                inputs += list(dict.fromkeys(got)); gen_ok = True; st["gen_cached"] = True; break
            continue
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=600, headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, json={
                "model": GEN_MODEL, "max_tokens": 16000, "provider": {"data_collection": "deny"},
                "messages": [{"role": "system", "content": GEN_PROMPT},
                             {"role": "user", "content": f"Problem:\n\n{text}"}]}).json()
            if "choices" not in r:
                st["gen_error"] = str(r.get("error", r))[:200]; continue
            msg = r["choices"][0]["message"]["content"]
            st["gen_usage"] = r.get("usage", {})
        except Exception as e:  # noqa: BLE001
            st["gen_error"] = str(e)[:200]; continue
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", msg or "", re.S)
        if len(blocks) < 2:
            continue
        (w / "gen.py").write_text(blocks[0]); (w / "validate.py").write_text(blocks[1])
        got = []
        for mode, k in (("small", 16), ("medium", 10), ("large", 4), ("edge", 10)):
            for s in range(k):
                rc, out, _ = _sh([sys.executable, str(w / "gen.py"), str(1000 * attempt + s), mode], timeout=20)
                if rc == 0 and out and out.strip() and len(out) < 6_000_000:
                    if _sh([sys.executable, str(w / "validate.py")], out, timeout=30)[0] == 0:
                        got.append(out)
        if len(set(got)) >= 15:
            inputs += list(dict.fromkeys(got)); gen_ok = True
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "gen.py").write_text(blocks[0]); (cache / "validate.py").write_text(blocks[1])
            break
    st["n_generated"] = len(inputs) - len(samples)
    if not gen_ok:
        return {**st, "status": "generator_failed"}
    # label: all working refs must agree (C++ preferred; Python refs only on inputs < 200 KB)
    tl = raw["time_limit"]
    tests, disagree = [], 0
    for x in inputs:
        outs = []
        for rf in refs:
            if rf["lang"] == "py" and len(x) > 200_000:
                continue
            rc, o, _ = _sh(rf["cmd"], x, timeout=max(10, 5 * tl))
            outs.append(o if rc == 0 else None)
        good = [o for o in outs if o is not None]
        if not good:
            continue
        if any(not same_output("stdin", good[0], o) for o in good[1:]):
            disagree += 1; continue
        tests.append({"input": x, "output": good[0], "testtype": "stdin"})
    st.update(n_tests=len(tests), ref_disagree=disagree)
    if len(tests) < 20 or disagree > 0.1 * len(inputs):
        return {**st, "status": "too_few_tests" if len(tests) < 20 else "refs_disagree"}
    # strength: kill rate on sample-passing mutants of the first C++ reference
    cpp = next((rf for rf in refs if rf["lang"] == "cpp"), None)
    kill = kept = 0
    if cpp:
        base = _strip_comments(Path(cpp["cmd"][0] + ".cpp").read_text())
        for j, mc in enumerate(_mutants(base, 12, random.Random(hash(task) & 0xffff))):
            (w / f"m{j}.cpp").write_text(mc)
            if _sh(["g++", "-std=gnu++20", "-O2", "-I/opt/ac-library", "-o", str(w / f"m{j}"), str(w / f"m{j}.cpp")], timeout=120)[0]:
                continue
            if not all(same_output("stdin", so, _sh([str(w / f"m{j}")], si, timeout=5)[1]) for si, so in samples):
                continue
            kept += 1
            for t in tests:
                rc, o, _ = _sh([str(w / f"m{j}")], t["input"], timeout=max(5, 3 * tl))
                if rc != 0 or not same_output("stdin", t["output"], o):
                    kill += 1; break
    st.update(mutants_kept=kept, mutants_killed=kill)
    if kept >= 3 and kill / kept < 0.8:
        return {**st, "status": "weak_tests"}
    # emit Harbor task (LiveCodeBench layout; agent sees ONLY public samples)
    pub = [{"input": si, "output": so, "testtype": "stdin"} for si, so in samples]
    priv = [t for t in tests if t["input"] not in {s for s, _ in samples}]
    rec = {"question_title": raw["title"], "question_content": text, "platform": "atcoder", "question_id": task,
           "contest_id": raw["contest"], "contest_date": raw["date"], "starter_code": "",
           "difficulty": {"a": "easy", "b": "easy", "c": "medium", "d": "medium"}.get(raw["letter"], "hard"),
           "public_test_cases": json.dumps(pub), "private_test_cases": _decode_private(priv), "metadata": "{}"}
    td = Path(f"{V}/{window}/tasks/{task}")
    (td / "environment" / "tests").mkdir(parents=True, exist_ok=True)
    (td / "tests").mkdir(parents=True, exist_ok=True)
    (td / "solution").mkdir(parents=True, exist_ok=True)
    (td / "task.toml").write_text(tmpl["task.toml"].replace("Naman Jain", "Millennium Harness FreshCP")
                                  .replace('timeout_sec = 180.0', 'timeout_sec = 900.0'))
    (td / "instruction.md").write_text(tmpl["instruction"].format(title=rec["question_title"], content=text))
    (td / "environment" / "Dockerfile").write_text(tmpl["Dockerfile"])
    (td / "environment" / "check_solution.py").write_text(tmpl["check_solution.py"])
    (td / "environment" / "tests" / "config.json").write_text(json.dumps({**rec, "private_test_cases": ""}))
    (td / "tests" / "config.json").write_text(json.dumps(rec))
    (td / "tests" / "final_test.py").write_text(re.sub(r"SUBPROCESS_TIMEOUT = \d+",
                                                       f"SUBPROCESS_TIMEOUT = {int(max(6, 5 * tl))}", tmpl["final_test.py"]))
    (td / "tests" / "test.sh").write_text(tmpl["test.sh"].replace(
        '[ -f solution.py ] || { echo "solution.py not found after 3 attempts"; exit 1; }',
        '[ -f solution.py ] || { echo "solution.py not found after 3 attempts"; mkdir -p /logs/verifier; '
        'echo 0 > /logs/verifier/reward.txt; exit 0; }'))
    os.chmod(td / "tests" / "test.sh", 0o755)
    py = next((rf for rf in refs if rf["lang"] == "py"), None)
    sol = Path(py["cmd"][1]).read_text() if py else ""
    (td / "solution" / "solve.sh").write_text(f"cat > /app/solution.py << 'EOF'\n{sol}\nEOF")
    vol.commit()
    return {**st, "status": "ok", "n_private": len(priv), "oracle_py": bool(py)}


# ------------------------------------------------------------------ local entry points

def _contests(window):
    import datetime as dt
    cs = json.loads((ROOT / "runs" / "freshcp" / "contests.json").read_text())
    lo, hi = (dt.datetime.fromisoformat(x).replace(tzinfo=dt.UTC) for x in WINDOWS[window])
    out = []
    for c in cs:
        d = dt.datetime.fromtimestamp(c["start_epoch_second"], dt.UTC)
        if lo <= d < hi and c["id"][:3] in LETTERS:
            out.append({"id": c["id"], "date": d.isoformat()})
    return sorted(out, key=lambda c: c["date"])


def _template():
    t = ROOT / "runs" / "tasks" / "livecodebench" / "abc329_f"
    ins = (t / "instruction.md").read_text()
    head, tail = ins.split("=== YOUR TASK ===", 1)
    return {"task.toml": (t / "task.toml").read_text(),
            "instruction": "=== PROBLEM ===\n\nTitle: {title}\n\n{content}\n\n=== YOUR TASK ===" + tail.replace("{", "{{").replace("}", "}}"),
            "Dockerfile": (t / "environment" / "Dockerfile").read_text(),
            "check_solution.py": (t / "environment" / "check_solution.py").read_text(),
            "final_test.py": (t / "tests" / "final_test.py").read_text(),
            "test.sh": (t / "tests" / "test.sh").read_text()}


def main():
    cmd, window = sys.argv[1], sys.argv[2]
    out = ROOT / "runs" / "freshcp"
    if cmd == "scrape":
        cs = _contests(window)
        print(f"{window}: {len(cs)} contests")
        with app.run():
            print(scrape_remote.remote(window, cs))
    elif cmd == "build":
        tasks = [e.path.split("/")[-1][:-5] for e in vol.listdir(f"/{window}/raw")]
        tmpl = _template()
        res = []
        with app.run():
            for r in build_remote.map([window] * len(tasks), tasks, [tmpl] * len(tasks), return_exceptions=True):
                res.append(r if isinstance(r, dict) else {"status": f"exception:{repr(r)[:120]}"})
        (out / f"build_{window}.json").write_text(json.dumps(res))
        from collections import Counter
        usd = sum((r.get("gen_usage") or {}).get("cost", 0) or 0 for r in res)
        with open(ROOT / "runs" / "ledger.jsonl", "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "item": f"freshcp-gen-{window}", "usd": round(usd, 4)}) + "\n")
        print(Counter(r["status"] for r in res), f"generator cost ${usd:.3f}")
    elif cmd == "stats":
        res = json.loads((out / f"build_{window}.json").read_text())
        from collections import Counter
        ok = [r for r in res if r["status"] == "ok"]
        print(Counter(r["status"] for r in res))
        print("ok by letter:", Counter(r["letter"] for r in ok))
        if ok:
            print("median private tests:", sorted(r["n_private"] for r in ok)[len(ok) // 2],
                  "| mutant kill:", sum(r["mutants_killed"] for r in ok), "/", sum(r["mutants_kept"] for r in ok),
                  "| python oracle:", sum(r["oracle_py"] for r in ok))
    elif cmd == "fetch":
        dest = Path(sys.argv[3])
        dest.mkdir(parents=True, exist_ok=True)
        os.system(f"modal volume get sorryless-sealed /{window}/tasks {dest} --force")


if __name__ == "__main__":
    main()
