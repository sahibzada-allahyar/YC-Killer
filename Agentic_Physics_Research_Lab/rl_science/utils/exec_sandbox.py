from __future__ import annotations
import subprocess, tempfile, os, shutil, textwrap, uuid, sys, pathlib

def run_python_code(code: str, workdir: str = ".", timeout: int = 60):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    rundir = tempfile.mkdtemp(prefix=run_id, dir=workdir)
    plots_dir = os.path.join(rundir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    script = os.path.join(rundir, "script.py")
    # force plots to land here
    code = "import os\nos.makedirs('plots', exist_ok=True)\n" + code
    pathlib.Path(script).write_text(code, encoding="utf-8")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run([sys.executable, "-u", script],
                          cwd=rundir, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, text=True)
    artifacts = {
        "plots": [os.path.join(rundir, "plots", f) for f in os.listdir(plots_dir)]
                 if os.path.isdir(plots_dir) else []
    }
    # copy plots to repo-level plots/
    dest = os.path.join(workdir, "plots")
    os.makedirs(dest, exist_ok=True)
    for p in artifacts["plots"]:
        shutil.copy2(p, os.path.join(dest, os.path.basename(p)))
    return proc.stdout, proc.stderr, {"plots": [os.path.join("plots", os.path.basename(p)) for p in artifacts["plots"]]}
