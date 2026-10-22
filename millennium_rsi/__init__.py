"""Millennium RSI: recursive self-improvement with a reward that cannot be gamed.

A coding agent improves its own harness (its strategy and a library of lemmas) one generation at a time. A change is
kept only if Millennium Harness certifies more held-out proofs with it: a kernel-checked proof of a frozen statement,
not an exit code.
"""

__version__ = "0.1.0"
