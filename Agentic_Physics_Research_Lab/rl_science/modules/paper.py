from __future__ import annotations
from typing import Tuple, Dict, Any, List
from ..utils.latex import build_minimal_tex

def write(ctx: Dict[str,Any]) -> Tuple[str, str]:
    title = "Autonomous Research Report"
    abs_ = "This draft was assembled by the ScientificEnv pipeline."
    intro = "We describe the data and motivation provided in the input."
    methods = ctx.get("methods","")
    results = ctx.get("results","")
    figs: List[str] = ctx.get("plots", [])

    tex_str = build_minimal_tex(
        title=title, abstract=abs_, intro=intro, methods=methods, results=results, figures=figs
    )
    # Save .tex. External compilation optional (pdflatex/latexmk if available)
    with open("paper.tex","w", encoding="utf-8") as f:
        f.write(tex_str)
    pdf_path = None  # compile offline if CI supports
    return ("paper.tex", pdf_path)
