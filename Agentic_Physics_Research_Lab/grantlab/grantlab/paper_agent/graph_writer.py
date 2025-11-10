from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
from .paper_graph import PaperGraph, Node

def _load_artifacts(run_dir: Path):
    res = {}
    for fn in ["hypothesis.json","plan.json","result.json","interpretation.json","lean_summary.json","data.json","evo_model.json","bayes.json"]:
        p = run_dir / fn
        if p.exists():
            res[fn] = json.loads(p.read_text(encoding="utf-8"))
    return res

def _make_graph(arts: dict, title: str) -> PaperGraph:
    g = PaperGraph()
    g.add(Node("title","title", content=f"# {title}

_Auto-generated on {datetime.now().isoformat(timespec='seconds')}_"))
    g.add(Node("abstract","abstract", content="We present a physics copilot pipeline integrating formal Lean checks, Bayesian inference, and AlphaEvolve-style symbolic search."))
    g.add(Node("intro","section", content="We outline goals, related work, and experimental setup."))
    g.add(Node("methods","section", content="Data, hypothesis family selection, experiment design, and Bayesian inference method."))
    g.add(Node("results","section", content="Key quantitative results, error bars, Lean checks, and evolved models."))
    g.add(Node("discussion","section", content="Limitations, delegations to external solvers, and future work."))
    g.add(Node("appendix","section", content="Formal invariants verified in Lean and additional figures."))

    g.link("title","abstract"); g.link("title","intro"); g.link("title","methods")
    g.link("title","results"); g.link("title","discussion"); g.link("title","appendix")

    # inject content from artifacts (simple)
    hyp = arts.get("hypothesis.json", {})
    plan = arts.get("plan.json", {})
    res  = arts.get("result.json", {})
    interp = arts.get("interpretation.json", {})
    lean = arts.get("lean_summary.json", {})
    evo = arts.get("evo_model.json", {})
    bayes = arts.get("bayes.json", {})

    g.nodes["methods"].content += f"

**Hypothesis:** `{hyp}`

**Plan:** `{plan}`
"
    g.nodes["results"].content += f"

**Summary stats:** `{res}`

**Interpretation:** `{interp}`
"
    g.nodes["results"].content += f"

**Evolved model:** `{evo}`

**Posterior summary:** keys={list(bayes.keys())}
"
    g.nodes["appendix"].content += f"

**Lean summary:** `{lean}`
"

    return g

def write_paper_from_run(run_dir: Path) -> Path:
    arts = _load_artifacts(run_dir)
    title = "Physics Copilot Report"
    g = _make_graph(arts, title)
    # linearize to markdown
    md = []
    order = ["title","abstract","intro","methods","results","discussion","appendix"]
    for k in order:
        n = g.nodes[k]
        md.append(n.content)
    out = run_dir / "paper.md"
    out.write_text("

".join(md), encoding="utf-8")
    g.to_json(run_dir / "paper_graph.json")
    return out
