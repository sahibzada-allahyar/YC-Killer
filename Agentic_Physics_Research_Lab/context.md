Awesome—here’s a complete, end‑to‑end repository you can drop into a folder and run.
It includes:

* A **modular pipeline** with the 5 required stages: **Hypothesis → Design Experiments → Run → Interpret → Write Report**.
* A **separate, modular Lean proof‑checking agent** (`grantlab/lean_agent/`) that generates Lean code, invokes Lean (if installed), and reports formal verification results of simple experiment invariants.
* A CLI: `grantlab run examples/simple_experiment/config.yml` to execute the full flow.
* Minimal example data + config + report template.
* Tests and a Makefile.

> **Note on Lean**: The pipeline works without Lean. If Lean 4 is installed (`lean --version` available), the Lean agent will formally check basic invariants (e.g., group counts add up) and include results in the report. The Lean agent is fully modular and can be extended to richer formalizations later.

---

## Repository tree

```
grantlab/
├─ README.md
├─ LICENSE
├─ Makefile
├─ pyproject.toml
├─ requirements.txt
├─ grantlab/
│  ├─ __init__.py
│  ├─ cli.py
│  ├─ config.py
│  ├─ pipeline.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ data.py
│  │  ├─ io.py
│  │  └─ logging.py
│  ├─ hypothesis/
│  │  ├─ __init__.py
│  │  └─ generator.py
│  ├─ design/
│  │  ├─ __init__.py
│  │  └─ experiment_designer.py
│  ├─ runner/
│  │  ├─ __init__.py
│  │  └─ executor.py
│  ├─ analysis/
│  │  ├─ __init__.py
│  │  └─ interpreter.py
│  ├─ reporting/
│  │  ├─ __init__.py
│  │  ├─ report_writer.py
│  │  └─ templates/
│  │     ├─ report.md.j2
│  │     └─ report.latex.j2
│  └─ lean_agent/
│     ├─ __init__.py
│     ├─ lean_agent.py
│     └─ templates/
│        └─ HypothesisProof.lean.j2
├─ examples/
│  └─ simple_experiment/
│     ├─ dataset.csv
│     └─ config.yml
└─ tests/
   ├─ test_pipeline.py
   └─ test_lean_agent.py
```

---

## Files

> Copy all of the following into files with matching paths.

### `README.md`
- [x] Copied

### `LICENSE`
- [x] Copied

### `Makefile`
- [x] Copied

### `pyproject.toml`
- [x] Copied

### `requirements.txt`
- [x] Copied

------

### `grantlab/__init__.py`
- [x] Copied

### `grantlab/cli.py`
- [x] Copied

### `grantlab/config.py`
- [x] Copied

### `grantlab/pipeline.py`
- [x] Copied

### `grantlab/utils/__init__.py`
- [x] Copied

### `grantlab/utils/data.py`
- [x] Copied

### `grantlab/utils/io.py`
- [x] Copied

### `grantlab/utils/logging.py`
- [x] Copied

---

### `grantlab/hypothesis/__init__.py`
- [x] Copied

### `grantlab/hypothesis/generator.py`
- [x] Copied

---

### `grantlab/design/__init__.py`
- [x] Copied

### `grantlab/design/experiment_designer.py`
- [x] Copied

---

### `grantlab/runner/__init__.py`
- [x] Copied

### `grantlab/runner/executor.py`
- [x] Copied

---

### `grantlab/analysis/__init__.py`
- [x] Copied

### `grantlab/analysis/interpreter.py`
- [x] Copied

---

### `grantlab/reporting/__init__.py`
- [x] Copied

### `grantlab/reporting/report_writer.py`
- [x] Copied

### `grantlab/reporting/templates/report.md.j2`
- [x] Copied

### `grantlab/reporting/templates/report.latex.j2`
- [x] Copied

---

### `grantlab/lean_agent/__init__.py`
- [x] Copied

### `grantlab/lean_agent/lean_agent.py`
- [x] Copied

### `grantlab/lean_agent/templates/HypothesisProof.lean.j2`
- [x] Copied

---

---

### `examples/simple_experiment/dataset.csv`
- [x] Copied

### `examples/simple_experiment/config.yml`
- [x] Copied

---

---

### `tests/test_pipeline.py`
- [x] Copied

### `tests/test_lean_agent.py`
- [x] Copied


---

## What this delivers for your grant requirements

* **Hypothesis**: `grantlab/hypothesis/generator.py` infers/constructs a testable mean-difference hypothesis from config + data.
* **Design Experiments**: `grantlab/design/experiment_designer.py` produces an experiment plan (allocation, n/group, test).
* **Run**: `grantlab/runner/executor.py` performs assignment (if needed) and executes the statistical test (Welch’s t by default).
* **Interpret**: `grantlab/analysis/interpreter.py` turns stats into a decision with rationale.
* **Write the Report**: `grantlab/reporting/report_writer.py` renders a complete Markdown report from a Jinja2 template.
* **Separate Lean Proof Agent**: `grantlab/lean_agent/lean_agent.py` + Lean template emits/compiles Lean code; results are folded into the report.

---

## Next extensions (optional)

* Add power analysis (e.g., via `statsmodels`) to auto‑choose `n_per_group`.
* Extend Lean template to import Mathlib and formalize properties of Welch’s t-test assumptions.
* Add plotting (saved PNGs referenced by the report).
* Support stratified assignment and covariate adjustment (ANCOVA).

---

If you want, I can tailor the hypothesis generator (e.g., multi-metric scanning, directional heuristics) or extend the Lean proofs to align with a specific theoretical model you have in mind.
