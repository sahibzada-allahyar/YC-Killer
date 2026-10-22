"""Minimal chat-completion clients (standard library only).

Models with a provider prefix ("z-ai/glm-5.3-flash", "deepseek/...") go through OpenRouter (OPENROUTER_API_KEY);
bare Claude model ids ("claude-haiku-5-5") go through the Anthropic Messages API (ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost += other.cost


@dataclass
class LLM:
    model: str
    max_tokens: int = 8000
    temperature: float = 0.7
    timeout: int = 600
    usage: Usage = field(default_factory=Usage)

    @property
    def provider(self) -> str:
        return "openrouter" if "/" in self.model else "anthropic"

    def complete(self, system: str, user: str) -> str:
        for attempt in range(4):
            try:
                return self._openrouter(system, user) if self.provider == "openrouter" else self._anthropic(system, user)
            except (urllib.error.URLError, TimeoutError, KeyError) as e:
                if attempt == 3:
                    raise RuntimeError(f"{self.model}: {e}") from e
                time.sleep(2 ** attempt * 3)
        raise AssertionError("unreachable")

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json",
                                                                                  **headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if 400 <= e.code < 500 and e.code != 429:
                raise ValueError(f"HTTP {e.code}: {body}") from e  # not retried: the request itself is wrong
            raise

    def _openrouter(self, system: str, user: str) -> str:
        d = self._post("https://openrouter.ai/api/v1/chat/completions",
                       {"authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                       {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                        "usage": {"include": True},
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        u = d.get("usage") or {}
        self.usage.add(Usage(1, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), float(u.get("cost") or 0)))
        return d["choices"][0]["message"]["content"] or ""

    def _anthropic(self, system: str, user: str) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ["ANTHROPIC_API_KEY_2"]
        d = self._post("https://api.anthropic.com/v1/messages",
                       {"x-api-key": key, "anthropic-version": "2023-06-01"},
                       {"model": self.model, "max_tokens": self.max_tokens,  # Claude 5 models fix their own sampling
                        "system": system, "messages": [{"role": "user", "content": user}]})
        u = d.get("usage") or {}
        self.usage.add(Usage(1, u.get("input_tokens", 0), u.get("output_tokens", 0)))
        return "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text")


class ScriptedLLM(LLM):
    """Replays canned replies: the first reply whose trigger occurs in the system or user prompt (tests, demos)."""

    def __init__(self, replies: list[tuple[str, str]]):
        super().__init__(model="scripted")
        self.replies = replies
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        self.usage.add(Usage(1))
        for trigger, reply in self.replies:
            if trigger in system + "\n" + user:
                return reply
        return ""
