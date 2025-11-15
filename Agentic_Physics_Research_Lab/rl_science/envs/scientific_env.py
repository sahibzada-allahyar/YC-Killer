from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple, List
import json
import time
from ..orchestrator import Orchestrator, OrchestratorConfig, Role, SubTask
from ..evaluators.rewards import compute_rewards
from ..modules import idea, literature, methods, analysis, paper, review, keywords

class Action(Enum):
    CALL_IDEA_MAKER = auto()
    CALL_IDEA_HATER = auto()
    UPDATE_PLAN = auto()
    RUN_METHODS = auto()
    RUN_ANALYSIS = auto()
    WRITE_PAPER = auto()
    RUN_REVIEW = auto()
    INSTALL_MISSING = auto()
    TERMINATE = auto()

@dataclass
class EnvState:
    step: int = 0
    plan: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    last_errors: List[str] = field(default_factory=list)
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

class ScientificEnv:
    """
    Minimal RL-style environment exposing the research pipeline as an MDP.
    Aligns with Denario's module graph (Idea → Literature → Methods → Analysis → Paper → Review).
    Observations are light JSON-like dicts. Rewards come from evaluators.rewards.
    """
    def __init__(self, max_steps: int = 60):
        self.max_steps = max_steps
        self.state = EnvState()
        self.ctx: Dict[str, Any] = {
            "data_description": "",
            "idea": None,
            "methods": None,
            "results": None,
            "plots": [],
            "paper_tex": None,
            "paper_pdf": None,
            "referee_md": None,
            "keywords": None,
            # run logs
            "stdout": "",
            "stderr": "",
        }
        self._orchestrator = self._make_orchestrator()

    def _make_orchestrator(self) -> Orchestrator:
        def planner(input_text: str) -> List[SubTask]:
            # Simple default plan: Idea → Methods → Analysis → Paper → Review
            return [
                SubTask("generate_idea", Role.PLANNER, ["IdeaMaker: 5 ideas", "IdeaHater critique", "select+improve"]),
                SubTask("design_methods", Role.RESEARCHER, ["Write methods.md ~500 words"]),
                SubTask("run_analysis", Role.ENGINEER, ["execute code", "save plots", "print stats"]),
                SubTask("write_paper", Role.RESEARCHER, ["assemble sections", "insert figures", "draft LaTeX"]),
                SubTask("review_paper", Role.RESEARCHER, ["referee report"]),
            ]

        def plan_reviewer(steps: List[SubTask], input_text: str) -> List[SubTask]:
            # Keep it simple: enforce <= 6 steps and ensure 'run_analysis' precedes 'write_paper'
            filtered = [s for s in steps if s.name in {"generate_idea","design_methods","run_analysis","write_paper","review_paper"}]
            names = [s.name for s in filtered]
            if "run_analysis" in names and "write_paper" in names:
                if names.index("run_analysis") > names.index("write_paper"):
                    # swap
                    ia, ip = names.index("run_analysis"), names.index("write_paper")
                    filtered[ia], filtered[ip] = filtered[ip], filtered[ia]
            return filtered[:6]

        executors = {
            Role.PLANNER: self._exec_idea_pipeline,
            Role.RESEARCHER: self._exec_researcher,
            Role.ENGINEER: self._exec_engineer,
            Role.TERMINATOR: self._exec_terminator,
        }
        return Orchestrator(planner, plan_reviewer, executors, OrchestratorConfig())

    # ===== EXECUTORS (align with paper's roles) =====
    def _exec_idea_pipeline(self, task: SubTask, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ideas = idea.generate_ideas(ctx.get("data_description",""))
        critiques = idea.critique_ideas(ideas, ctx.get("data_description",""))
        best = idea.select_and_improve(ideas, critiques)
        ctx["idea"] = best
        ctx["keywords"] = keywords.select_keywords(best, ctx.get("data_description",""))
        return {"idea": best, "keywords": ctx["keywords"]}

    def _exec_researcher(self, task: SubTask, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if task.name == "design_methods":
            m = methods.design(ctx.get("data_description",""), ctx.get("idea",""))
            ctx["methods"] = m
            return {"methods": m}
        elif task.name == "write_paper":
            tex, pdf = paper.write(ctx)
            ctx["paper_tex"], ctx["paper_pdf"] = tex, pdf
            return {"paper_tex": tex, "paper_pdf": pdf}
        elif task.name == "review_paper":
            rep = review.referee(ctx)
            ctx["referee_md"] = rep
            return {"referee_md": rep}
        return {}

    def _exec_engineer(self, task: SubTask, ctx: Dict[str, Any]) -> Dict[str, Any]:
        res = analysis.run(ctx.get("data_description",""), ctx.get("idea",""), ctx.get("methods",""))
        ctx.update(res)
        return res

    def _exec_terminator(self, task: SubTask, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["terminated"] = True
        return {"terminated": True}

    # ====== RL-style API ======
    def reset(self, data_description: str) -> Dict[str, Any]:
        self.state = EnvState()
        self.ctx.update({
            "data_description": data_description,
            "idea": None, "methods": None,
            "results": None, "plots": [], "paper_tex": None, "paper_pdf": None,
            "referee_md": None, "keywords": None,
            "stdout": "", "stderr": ""
        })
        self.state.plan = {"steps": [{"name": s.name, "agent": s.agent.name, "bullets": s.bullets}
                                     for s in self._orchestrator.build_plan(data_description).steps]}
        return self._observe()

    def step(self, action: Action) -> Tuple[Dict[str,Any], float, bool, Dict[str,Any]]:
        self.state.step += 1
        self.state.last_errors = []

        # Map high-level action to a subtask execution
        try:
            if action == Action.CALL_IDEA_MAKER:
                self._orchestrator.executors[Role.PLANNER](SubTask("generate_idea", Role.PLANNER, []), self.ctx)
            elif action == Action.RUN_METHODS:
                self._exec_researcher(SubTask("design_methods", Role.RESEARCHER, []), self.ctx)
            elif action == Action.RUN_ANALYSIS:
                self._exec_engineer(SubTask("run_analysis", Role.ENGINEER, []), self.ctx)
            elif action == Action.WRITE_PAPER:
                self._exec_researcher(SubTask("write_paper", Role.RESEARCHER, []), self.ctx)
            elif action == Action.RUN_REVIEW:
                self._exec_researcher(SubTask("review_paper", Role.RESEARCHER, []), self.ctx)
            elif action == Action.TERMINATE:
                self._exec_terminator(SubTask("terminate", Role.TERMINATOR, []), self.ctx)
                self.state.done = True
        except Exception as e:
            self.state.last_errors.append(str(e))

        obs = self._observe()
        reward = compute_rewards(self.ctx, self.state.last_errors)
        done = self.state.done or self.state.step >= self.max_steps
        info = {"errors": self.state.last_errors}
        return obs, reward, done, info

    def _observe(self) -> Dict[str, Any]:
        return {
            "step": self.state.step,
            "have_idea": self.ctx["idea"] is not None,
            "have_methods": self.ctx["methods"] is not None,
            "have_results": self.ctx.get("results") is not None,
            "n_plots": len(self.ctx.get("plots", [])),
            "paper_ready": self.ctx["paper_tex"] is not None,
            "review_ready": self.ctx["referee_md"] is not None,
            "keywords": self.ctx.get("keywords"),
        }
