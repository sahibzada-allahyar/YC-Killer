from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Callable
import time
import uuid

class Role(Enum):
    PLAN_SETTER = auto()
    PLANNER = auto()
    PLAN_REVIEWER = auto()
    CONTROLLER = auto()
    ENGINEER = auto()
    RESEARCHER = auto()
    INSTALLER = auto()
    TERMINATOR = auto()

@dataclass
class SubTask:
    name: str
    agent: Role
    bullets: List[str]

@dataclass
class Plan:
    steps: List[SubTask] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class StepStatus:
    idx: int
    subtask: SubTask
    status: str  # 'pending'|'running'|'done'|'failed'
    artifacts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

@dataclass
class OrchestratorConfig:
    nrounds: int = 500
    nfails: int = 5

class Orchestrator:
    """
    Minimal Planning & Control orchestrator.
    - Builds a plan using planner + plan reviewer dialogue (single round by default, as in paper)
    - Executes step-by-step with a controller and bounded retries.
    """
    def __init__(
        self,
        planner: Callable[[str], List[SubTask]],
        plan_reviewer: Callable[[List[SubTask], str], List[SubTask]],
        executors: Dict[Role, Callable[[SubTask, Dict[str, Any]], Dict[str, Any]]],
        cfg: OrchestratorConfig = OrchestratorConfig(),
        nreviews: int = 1,  # aligns with paper defaults
        nsteps_cap: int = 8 # typical range 3-8 per paper
    ):
        self.planner = planner
        self.plan_reviewer = plan_reviewer
        self.executors = executors
        self.cfg = cfg
        self.nreviews = nreviews
        self.nsteps_cap = nsteps_cap
        self._rounds = 0
        self._failures = 0

    def build_plan(self, input_text: str) -> Plan:
        steps = self.planner(input_text)[: self.nsteps_cap]
        for _ in range(self.nreviews):
            steps = self.plan_reviewer(steps, input_text)[: self.nsteps_cap]
        return Plan(steps=steps)

    def run(self, plan: Plan, context: Dict[str, Any]) -> List[StepStatus]:
        log: List[StepStatus] = []
        for i, sub in enumerate(plan.steps):
            self._assert_hard_limits()
            status = StepStatus(idx=i, subtask=sub, status="running")
            try:
                if sub.agent not in self.executors:
                    raise RuntimeError(f"No executor for role {sub.agent}")
                artifacts = self.executors[sub.agent](sub, context)
                status.status = "done"
                status.artifacts = artifacts or {}
                self._failures = 0  # reset on success
            except Exception as e:
                status.status = "failed"
                status.errors.append(str(e))
                self._failures += 1
                if self._failures >= self.cfg.nfails:
                    # emulate 'terminator' behavior from paper
                    term = SubTask("terminate", Role.TERMINATOR, ["abort session"])
                    _ = self.executors.get(Role.TERMINATOR, lambda *_: {})(term, context)
                    log.append(status)
                    break
            finally:
                log.append(status)
                self._rounds += 1
                self._assert_hard_limits()
        return log

    def _assert_hard_limits(self):
        if self._rounds > self.cfg.nrounds:
            raise RuntimeError("Exceeded nrounds; aborting (loop guard).")
