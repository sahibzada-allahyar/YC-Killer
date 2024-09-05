"""
Jinja templates for both stages.

These strings are short so they remain maintainable in code.
"""
from textwrap import dedent

IDEA_PROMPT = dedent(
    """
    Come up with a simple crypto trading idea that can be represented
    as a mathematical formula. Keep it to three sentences.

    Return JSON:
      { "idea": "<three sentences>" }
    """
).strip()

ALPHA_PROMPT = dedent(
    """
    Your job is to convert the following idea into {{ n }} distinct alpha
    expressions.

    IDEA:
    {{ idea }}

    Rules:
    * Use ONLY these column names: {{ cols }}
    * Use ONLY these transforms:  {{ transforms }}
    * Prefix notation like:      ts_zscore(div(sub(high, low), close), 30)
    * Return **JSON List** of dicts like:
        { "frequency": "1h", "alpha": "..." }

    NO commentary, NO markdown, NO code fences.
    """
).strip()
