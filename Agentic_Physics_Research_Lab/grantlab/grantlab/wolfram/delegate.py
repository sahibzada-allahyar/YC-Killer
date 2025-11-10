from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class TaskRequest:
    task_type: str  # "integral"|"symbolic_integral"|"ode"|"pde"|"special_function"|"simplify"|"eigen"|"groebner"|"laplace"|"series"|"algebraic_solve"|"large_linear"
    description: str
    size_hint: Optional[int] = None  # e.g., matrix dimension
    stiffness_hint: bool = False

# What WA is (typically) better at than an LLM:
WA_BETTER = {
    "symbolic_integral",    # Risch / algebraic decision procedures
    "laplace",
    "series",
    "groebner",
    "algebraic_solve",
    "special_function",     # high-precision gamma/zeta/elliptic integrals, etc.
    "pde",                  # general PDE symbolic/numeric solvers
    "eigen",                # large/sparse eigenproblems
    "large_linear",         # large linear systems / matrix functions
    "simplify",             # exact simplification / factorization
}

def should_delegate_to_wolfram(req: TaskRequest) -> bool:
    if req.task_type in WA_BETTER:
        return True
    if req.task_type == "ode" and (req.stiffness_hint or (req.size_hint and req.size_hint > 3)):
        return True
    if req.task_type == "integral" and (req.size_hint and req.size_hint > 1):  # multi-dimensional/quadrature
        return True
    return False
