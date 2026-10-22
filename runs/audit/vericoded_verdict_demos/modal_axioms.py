"""Faithful `#print axioms` probes on published vericoded Lean files (Mathlib image of bench/audit/vericoded_rung.py)."""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
import modal
from bench.audit import vericoded_rung as vr

VC = Path('/tmp/vc/vericoding/vericoded/lean')
PROBES = {
  'apps/gemini-2.5-pro/LA0288': ['solve_spec_satisfied', 'helper_mod_pos'],
  'apps/gemini-2.5-pro/LA0499': ['solve_spec_satisfied', 'int_to_string_output_nonempty'],
  'numpy-triple/claude-opus-4.1/LT0376': ['chebyshev_spec'],
  'numpy-triple/gpt-5/LT0376': ['chebyshev_spec'],
  'apps/claude-opus-4.1/LA0206': ['solve_spec_satisfied'],
  'apps/gemini-2.5-pro/LA0206': ['solve_spec_satisfied'],
  'verified-cogen/claude-opus-4.1/LJ0108': ['differenceMaxMin_spec_satisfied'],
  'verified-cogen/gemini-2.5-pro/LJ0108': ['differenceMaxMin_spec_satisfied'],
  'verified-cogen/gpt-5/LJ0108': ['differenceMaxMin_spec_satisfied'],
  'verified-cogen/grok-code/LJ0108': ['differenceMaxMin_spec_satisfied'],
  'apps/claude-sonnet-4/LA0189': ['solve_spec_satisfied'],
  'apps/gemini-2.5-flash/LA0189': ['solve_spec_satisfied'],
  'apps/glm-4.5/LA0189': ['solve_spec_satisfied'],
  'apps/gpt-5/LA0189': ['solve_spec_satisfied'],
  'bignum/gpt-5/LB0009': ['BignumLean.DivMod_spec'],
  'bignum/gpt-5/LB0004': ['BignumLean.Compare_spec', 'BignumLean.CompareUnequal_spec'],
  'bignum/gpt-5/LB0006': ['BignumLean.Compare_spec'],
  'bignum/gpt-5/LB0053': ['ModExp_int_spec'],
  'apps/claude-sonnet-4/LA0012': ['solve_spec_satisfied'],
  'apps/gpt-5/LA0583': ['solve_spec_satisfied'],
  'apps/gpt-5/LA0411': ['solve_spec_satisfied'],
}
app = modal.App('sorryless-audit-verdicts')

@app.function(image=vr.image, cpu=1, memory=3072, timeout=900, max_containers=24)
def probe(key: str, text: str) -> dict:
    from sorryless import lean as sl
    lean = sl.find_lean()
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'Solution.lean').write_text(text)
        t = time.time()
        p = subprocess.run([lean, *vr.LAKE_OPTIONS, 'Solution.lean'], cwd=d, env=sl._env(d), capture_output=True,
                           text=True, timeout=600)
        return {'key': key, 'rc': p.returncode, 'secs': round(time.time() - t, 1), 'out': (p.stdout + p.stderr)[-6000:]}

@app.local_entrypoint()
def main():
    jobs = []
    for k, names in PROBES.items():
        text = (VC / f'{k}.lean').read_text()
        text = text.rstrip('\n') + '\n\n' + '\n'.join(f'#print axioms {n}' for n in names) + '\n'
        jobs.append((k, text))
    out = Path(os.environ['OUT'])
    with out.open('w') as f:
        for r in probe.starmap(jobs):
            f.write(json.dumps(r) + '\n'); print(r['key'], r['rc'], r['secs'])
