from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal, Callable

# Optional backends
try:
    import wolframalpha  # Wolfram|Alpha HTTP API
except Exception:
    wolframalpha = None

try:
    # Minimal MCP client; adapt to your MCP client if it differs
    # For example, if you run an MCP tool server named "wolfram"
    from mcp import Client as MCPClient  # type: ignore
except Exception:
    MCPClient = None

# Optional Wolfram Language (cloud) via wolframclient
try:
    from wolframclient.evaluation import WolframLanguageSession  # type: ignore
    from wolframclient.language import wl, wlexpr  # type: ignore
except Exception:
    WolframLanguageSession = None
    wl = None
    wlexpr = None


@dataclass
class Budget:
    """Per-session budgets."""
    usd: float
    latency_s: float
    tokens: int

    def charge(self, usd: float, latency: float, tokens: int) -> bool:
        self.usd -= usd
        self.latency_s -= latency
        self.tokens -= tokens
        return (self.usd >= 0.0) and (self.latency_s >= 0.0) and (self.tokens >= 0)


@dataclass
class ToolEstimate:
    """Estimates for a candidate call used by the delegation policy."""
    cost_usd: float
    latency_s: float
    tokens: int
    quality: float  # expected answer quality 0..1


class DelegationPolicy:
    """
    Budget- and quality-aware routing policy.
    - Rejects a tool call if it would exceed budgets.
    - Among feasible tools, chooses highest utility:  quality - λ_cost*cost - λ_lat*latency
    """
    def __init__(self,
                 budget: Budget,
                 lambda_cost: float = 1.0,
                 lambda_latency: float = 0.25,
                 min_quality: float = 0.3):
        self.budget = budget
        self.lambda_cost = lambda_cost
        self.lambda_latency = lambda_latency
        self.min_quality = min_quality

    def select(self, options: Dict[str, ToolEstimate]) -> Optional[str]:
        feasible = {}
        for name, est in options.items():
            # Hard budget cut
            if (self.budget.usd - est.cost_usd < 0 or
                self.budget.latency_s - est.latency_s < 0 or
                self.budget.tokens - est.tokens < 0):
                continue
            if est.quality < self.min_quality:
                continue
            utility = est.quality - self.lambda_cost * est.cost_usd - self.lambda_latency * est.latency_s
            feasible[name] = (utility, est)
        if not feasible:
            return None
        # argmax utility
        return max(feasible.items(), key=lambda kv: kv[1][0])[0]

    def charge(self, est: ToolEstimate) -> bool:
        return self.budget.charge(est.cost_usd, est.latency_s, est.tokens)


class WolframClient:
    """
    Unified Wolfram entry point with budget-aware delegation.

    Config:
      - MCP_WOLFRAM_ENDPOINT (optional): if set and MCP client available, use MCP.
      - WOLFRAM_ALPHA_APPID (optional): Wolfram|Alpha REST.
      - WOLFRAM_CLOUD_KERNEL (optional): path/url for cloud session (wolframclient).
    """
    def __init__(self,
                 budget: Budget,
                 policy: Optional[DelegationPolicy] = None,
                 mcp_endpoint: Optional[str] = None,
                 alpha_app_id: Optional[str] = None,
                 cloud_kernel: Optional[str] = None):
        self.policy = policy or DelegationPolicy(budget)
        self._mcp_endpoint = mcp_endpoint or os.getenv("MCP_WOLFRAM_ENDPOINT")
        self._alpha_app_id = alpha_app_id or os.getenv("WOLFRAM_ALPHA_APPID")
        self._cloud_kernel = cloud_kernel or os.getenv("WOLFRAM_CLOUD_KERNEL")

        self._alpha_client = None
        if self._alpha_app_id and wolframalpha is not None:
            self._alpha_client = wolframalpha.Client(self._alpha_app_id)

        self._mcp_client = None
        if self._mcp_endpoint and MCPClient is not None:
            # Your MCP client bootstrap here; pseudo-code:
            self._mcp_client = MCPClient(self._mcp_endpoint)  # type: ignore

        self._wl_session = None
        if self._cloud_kernel and WolframLanguageSession is not None:
            self._wl_session = WolframLanguageSession(self._cloud_kernel)

    def _estimate(self, backend: Literal["mcp", "alpha", "wl"], query: str) -> ToolEstimate:
        # Rough defaults; tune from telemetry
        if backend == "mcp":
            return ToolEstimate(cost_usd=0.01, latency_s=0.8, tokens=150, quality=0.85)
        if backend == "alpha":
            return ToolEstimate(cost_usd=0.002, latency_s=0.5, tokens=50, quality=0.70)
        if backend == "wl":
            return ToolEstimate(cost_usd=0.00, latency_s=1.2, tokens=120, quality=0.90)
        raise ValueError("unknown backend")

    def evaluate(self, query: str) -> Dict[str, Any]:
        """
        Evaluate a free-form query; returns {'backend': str, 'result': Any, 'raw': Any}.
        """
        candidates: Dict[str, ToolEstimate] = {}
        if self._mcp_client is not None:
            candidates["mcp"] = self._estimate("mcp", query)
        if self._alpha_client is not None:
            candidates["alpha"] = self._estimate("alpha", query)
        if self._wl_session is not None:
            candidates["wl"] = self._estimate("wl", query)

        if not candidates:
            raise RuntimeError("No Wolfram backend available. Set MCP_WOLFRAM_ENDPOINT, WOLFRAM_ALPHA_APPID or WOLFRAM_CLOUD_KERNEL.")

        choice = self.policy.select(candidates)
        if choice is None:
            raise RuntimeError("All candidate backends exceed budgets or fail quality threshold.")

        est = candidates[choice]
        t0 = time.time()
        if choice == "mcp":
            # Example MCP call; adjust to your tool name / schema.
            raw = self._mcp_client.call_tool("wolfram", {"query": query})  # type: ignore
            result = raw.get("text") if isinstance(raw, dict) else raw
        elif choice == "alpha":
            res = self._alpha_client.query(query)
            # Extract plaintext if possible; fall back to pods
            try:
                result = next(res.results).text
                raw = res
            except Exception:
                result, raw = str(res), res
        else:  # 'wl'
            expr = wlexpr(query) if wlexpr is not None else query
            result = self._wl_session.evaluate(expr)
            raw = result

        latency = time.time() - t0
        # Charge actual observed latency; cost/tokens remain estimate unless you meter them
        self.policy.charge(ToolEstimate(est.cost_usd, latency, est.tokens, est.quality))

        return {"backend": choice, "result": result, "raw": raw}
