import pathlib
import pytest
import shutil
import subprocess

@pytest.mark.optional
def test_lean_templates_exist():
    base = pathlib.Path("grantlab") / "lean" / "templates"
    if not base.exists():
        pytest.skip("lean templates folder not present")
    files = list(base.glob("*.lean"))
    assert files, "No .lean templates found; expected at least one (e.g., sho.lean)"

@pytest.mark.network
@pytest.mark.slow
def test_mathlib_build_or_skip():
    if shutil.which("lean") is None:
        pytest.skip("lean not found on PATH")
    try:
        subprocess.run(["lean", "--version"], check=True, capture_output=True)
    except Exception:
        pytest.skip("lean is installed, but failed to run; skipping")
    base = pathlib.Path("grantlab") / "lean" / "templates"
    files = list(base.glob("*.lean"))
    if not files:
        pytest.skip("No lean templates to compile")
    try:
        subprocess.run(["lean", str(files[0])], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        blob = ((e.stderr or b"") + (e.stdout or b"")).lower()
        if b"mathlib" in blob and b"unknown package" in blob:
            pytest.skip("Mathlib not installed; skipping compilation test")
        raise
