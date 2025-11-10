from __future__ import annotations
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from typing import Dict, Any, List

@dataclass
class Invariant:
    name: str
    ok: bool
    detail: str | None = None

def _env(template_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )

class LeanAgent:
    """
    Standalone Lean proof checking agent.
    - Renders a Lean file from a template and facts from the experiment.
    - Calls `lean --make` to check the file.
    - Returns a structured summary.
    """
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.template_dir = Path(__file__).parent / "templates"

    @staticmethod
    def _lean_available() -> bool:
        return shutil.which("lean") is not None

    def _render(self, params: Dict[str, Any]) -> Path:
        env = _env(self.template_dir)
        tmpl = env.get_template("HypothesisProof.lean.j2")
        code = tmpl.render(**params)
        out = self.work_dir / "HypothesisProof.lean"
        out.write_text(code, encoding="utf-8")
        return out

    def _run_lean(self, lean_file: Path) -> subprocess.CompletedProcess:
        # We simply check the file. If it compiles, proofs passed.
        cmd = ["lean", "--make", str(lean_file)]
        return subprocess.run(cmd, cwd=str(self.work_dir), capture_output=True, text=True)

    def verify(self, hypothesis: Dict[str, Any], plan: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        n_a, n_b = int(result["n_a"]), int(result["n_b"])
        n_total = n_a + n_b
        params = {
            "n_a": n_a,
            "n_b": n_b,
            "n_total": n_total,
        }
        lean_file = self._render(params)

        invs: List[Invariant] = [
            Invariant(name="counts_nonnegative", ok=(n_a >= 0 and n_b >= 0)),
            Invariant(name="total_matches_sum_python", ok=(n_total == n_a + n_b)),
        ]

        if not self._lean_available():
            return {
                "available": False,
                "success": False,
                "file": str(lean_file),
                "invariants": [inv.__dict__ for inv in invs],
                "stdout": "",
                "stderr": "lean not found on PATH",
            }

        proc = self._run_lean(lean_file)
        success = (proc.returncode == 0)
        # One more invariant: lean compiled
        invs.append(Invariant(name="lean_compilation", ok=success, detail="Lean accepted all proofs" if success else "Lean failed"))
        return {
            "available": True,
            "success": success,
            "file": str(lean_file),
            "invariants": [inv.__dict__ for inv in invs],
            "stdout": proc.stdout[-2000:],  # tail to avoid huge logs
            "stderr": proc.stderr[-2000:],
        }
