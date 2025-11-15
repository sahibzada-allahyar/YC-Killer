# ScientificEnv (Autonomous Science RL)

This module adds a Denario-inspired research pipeline to your RL environment:
Idea → Methods → Analysis → Paper → Review (Table 1; see paper). Planning & Control
orchestration with guardrails (`nrounds`, `nfails`) mirrors the reference design (Sec. 2.2.2).

**Rewards** incentivize: printing quantitative info, producing plots, writing methods (~500 words),
assembling a draft, and passing a referee check. Guardrails penalize dummy/synthetic data.

## Quick start
```bash
pip install -e .
pytest -q
````

## Extending

* Replace `modules/idea.py` with your LLM-backed IdeaMaker/IdeaHater.
* Implement `modules/literature.py` with your Semantic Scholar/Owl backend.
* Toggle LaTeX compilation in `modules/paper.py` if `latexmk` is present.
* Add adaptive planning by calling `orchestrator.build_plan()` between steps (Sec. 6.4).
