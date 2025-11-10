from pathlib import Path
from grantlab.pipeline import run_pipeline

def test_pipeline_runs(tmp_path: Path):
    # Copy example config into tmp and set output there
    cfg = tmp_path / "config.yml"
    cfg.write_text("""\
project: "CI Test"
dataset: "examples/simple_experiment/dataset.csv"
id_column: "id"
outcome_column: "conversion"
group_column: ""
group_names: ["control","treatment"]
hypothesis:
  type: "mean_difference"
  direction: "treatment > control"
  metric: "conversion"
  alpha: 0.2
design:
  assignment: "random"
  n_per_group: 20
  seed: 1
  test: "welch_t"
run:
  bootstrap_iters: 100
report:
  out_dir: "%s"
  format: "md"
  include_plots: false
""" % (tmp_path.as_posix()), encoding="utf-8")
    run_pipeline(cfg)
    # Expect a report in the out_dir
    outs = list((tmp_path).glob("run_*/report.md"))
    assert outs, "report.md not generated"
