# GrantLab — End-to-End Experimental Pipeline (with Lean Proof Agent)

GrantLab runs a full scientific/AB-testing pipeline:

1. **Hypothesis** generation (from data + config)
2. **Design Experiments** (sample size, allocation, test choice)
3. **Run** the experiments (randomization + analysis)
4. **Interpret** the results (p-values, effect sizes, decisions)
5. **Write the Report** (Markdown; optional LaTeX template included)

It also includes a **modular Lean 4 proof-checking agent** that:
- Generates Lean code from experiment invariants (e.g., group counts, disjointness),
- Invokes `lean` (if installed) to **formally check** those invariants,
- Returns a structured verification summary included in the report.

> If Lean is not installed, the rest of the pipeline still runs; the proof step is skipped gracefully.

---

## Quick start

```bash
# 1) Create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install python deps
pip install -r requirements.txt

# 3) (Optional) Install Lean 4
# See: https://leanprover.github.io/lean4/doc/quickstart.html
# On macOS/Linux (via elan):
# curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
# After that, ensure `lean --version` works.

# 4) Run the example end-to-end
python -m grantlab run examples/simple_experiment/config.yml
```

Outputs will be written to `examples/simple_experiment/outputs/` (report.md, artifacts, logs).

---

## CLI

```bash
# Show help
python -m grantlab --help

# Initialize a blank project config in ./my_project/config.yml
python -m grantlab init my_project

# Run an experiment from config
python -m grantlab run path/to/config.yml

# Run only Lean proof checking for a plan/result
python -m grantlab lean-check path/to/config.yml
```

---

## Configuration

See `examples/simple_experiment/config.yml` for a minimal config. Key fields:

```yaml
project: "Example AB Test"
dataset: "examples/simple_experiment/dataset.csv"
id_column: "id"
outcome_column: "conversion"
group_column: "group"      # if empty or missing, the runner will randomize groups
group_names: ["control", "treatment"]

hypothesis:
  type: "mean_difference"  # supports "mean_difference"
  direction: "treatment > control"
  metric: "conversion"
  alpha: 0.05

design:
  assignment: "random"     # "random" or "stratified" (by covariates)
  n_per_group: 20
  seed: 42
  test: "welch_t"          # "t" or "welch_t" or "mannwhitney"

run:
  bootstrap_iters: 1000

report:
  out_dir: "examples/simple_experiment/outputs"
  format: "md"             # "md" (Markdown). LaTeX template also provided.
  include_plots: false
```

---

## Lean Agent

* Template: `grantlab/lean_agent/templates/HypothesisProof.lean.j2`
* Python wrapper: `grantlab/lean_agent/lean_agent.py`

The agent:

1. Emits a Lean file parameterized by experiment counts and simple invariants.
2. Runs `lean --make <file>.lean`.
3. Parses success/exit status to determine “proved” vs “failed”.

Extendable to richer math by replacing the template and imported libraries.

---

## Tests

```bash
pytest
```

`test_lean_agent.py` is skipped automatically if `lean` is not available.

---

## License

MIT
