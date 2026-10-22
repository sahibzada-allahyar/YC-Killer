"""Lean tasks for self-improvement: the curated DEV split of the vericoding-derived pool (runs/fv/).

Only DEV tasks are used; the TEST split stays held out. Train and validation are a deterministic split by task id.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FV = ROOT / "runs" / "fv"


@dataclass(frozen=True)
class Task:
    id: str
    cell: str
    frozen: dict

    @property
    def skeleton(self) -> str:
        return self.frozen["skeleton"]


def _rank(tid: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{tid}".encode()).hexdigest()


def load(n_train: int, n_val: int, cells: tuple[str, ...] = ("L-fv", "L-apps", "L-np"), salt: str = "rsi-v1",
         fv: Path = FV) -> tuple[list[Task], list[Task]]:
    cur = json.loads((fv / "curated.json").read_text())["tasks"]
    ids = sorted((t for t, v in cur.items() if t.startswith("L") and v["part"] == "dev" and v["status"] == "OK"
                  and v["cell"] in cells and (fv / "frozen" / f"{t}.json").is_file()), key=lambda t: _rank(t, salt))
    if n_train + n_val > len(ids):
        raise ValueError(f"asked for {n_train}+{n_val} tasks, only {len(ids)} available")
    mk = lambda t: Task(t, cur[t]["cell"], json.loads((fv / "frozen" / f"{t}.json").read_text()))  # noqa: E731
    return [mk(t) for t in ids[:n_train]], [mk(t) for t in ids[n_train:n_train + n_val]]
