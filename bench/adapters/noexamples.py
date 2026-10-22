"""Exploratory (post-freeze): "no examples" variant of a LiveCodeBench-format Harbor dataset — the real-work regime.

Neither the agent nor the spec writer sees sample I/O: examples are stripped from the statement, check_solution.py and
the agent-visible config are removed, and the instruction no longer mentions sample tests. Hidden tests are unchanged.
  python bench/adapters/noexamples.py SRC_DIR DST_DIR
"""
import json
import re
import shutil
import sys
from pathlib import Path


def strip_examples(text: str) -> str:
    # LeetCode style: "Example 1: ... Constraints:" ; AtCoder style: "Sample Input 1 ..." to the end
    text = re.sub(r"\n\s*Example\s*1\s*:.*?(?=\n\s*Constraints\s*:?)", "\n", text, flags=re.S)
    text = re.sub(r"\n\s*Sample Input\s*1.*$", "\n", text, flags=re.S)
    return text.strip() + "\n"


INSTR_TAIL = """=== YOUR TASK ===
{task}
There are no test files: make sure your solution is correct and efficient for every input allowed by the constraints.
"""


def convert(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)
    cfg = json.loads((src / "tests" / "config.json").read_text())
    stmt = strip_examples(cfg["question_content"])
    head = (src / "instruction.md").read_text().split("=== YOUR TASK ===")[0]
    title = re.search(r"Title:\s*(.*)", head)
    starter = cfg.get("starter_code") or ""
    task = ("Complete the Solution class in the starter code and save it to solution.py." if starter else
            "Write a Python program that reads from standard input and writes to standard output; save it as "
            "solution.py.")
    parts = [f"=== PROBLEM ===\n\nTitle: {title.group(1) if title else ''}\n\n{stmt}"]
    if starter:
        parts.append(f"=== STARTER CODE ===\n{starter}\n")
    parts.append(INSTR_TAIL.format(task=task))
    (dst / "instruction.md").write_text("\n".join(parts))
    # agent sees no tests at all
    for f in ("check_solution.py",):
        (dst / "environment" / f).unlink(missing_ok=True)
    shutil.rmtree(dst / "environment" / "tests", ignore_errors=True)
    df = (dst / "environment" / "Dockerfile").read_text()
    df = re.sub(r"COPY check_solution.py .*\n", "", df)
    df = re.sub(r"COPY tests/config.json .*\n", "", df)
    (dst / "environment" / "Dockerfile").write_text(df)
    # the spec writer sees the same stripped statement and NO examples
    cfg_noex = dict(cfg, question_content=stmt, public_test_cases="[]", private_test_cases="")
    (dst / "tests" / "spec_view.json").write_text(json.dumps(cfg_noex))


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in sorted(src.iterdir()):
        if (t / "tests" / "config.json").exists():
            convert(t, dst / t.name)
            n += 1
    print(f"converted {n} tasks into {dst}")
