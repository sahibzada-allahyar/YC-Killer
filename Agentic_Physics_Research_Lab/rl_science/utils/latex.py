from __future__ import annotations
from typing import List

def build_minimal_tex(title: str, abstract: str, intro: str, methods: str, results: str, figures: List[str]) -> str:
    figs = "\n".join(
        [f"\\begin{{figure}}[t]\\centering\\includegraphics[width=0.85\\linewidth]{{{p}}}\\caption{{Auto figure: {i+1}}}\\end{{figure}}"
         for i, p in enumerate(figures)]
    )
    return rf"""
\documentclass[11pt]{{article}}
\usepackage{{graphicx}}
\usepackage[margin=1in]{{geometry}}
\title{{{title}}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{abstract}
\end{{abstract}}

\section*{{Introduction}}
{intro}

\section*{{Methods}}
{methods}

\section*{{Results}}
{results}

{figs}

\end{{document}}
""".strip()
