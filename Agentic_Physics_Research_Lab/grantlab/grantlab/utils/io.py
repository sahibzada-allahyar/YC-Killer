from __future__ import annotations
from pathlib import Path
from datetime import datetime

def ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d

def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

class TimestampedPath:
    def __init__(self, root: Path, prefix: str = "run_"):
        self.root = root
        self.prefix = prefix
    def make_dir(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        p = self.root / f"{self.prefix}{ts}"
        p.mkdir(parents=True, exist_ok=True)
        return p
