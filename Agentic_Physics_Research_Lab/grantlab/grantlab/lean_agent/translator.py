from __future__ import annotations
import re
from typing import Optional

MATH_INLINE = re.compile(r"\$(.+?)\$")
MATH_BLOCK = re.compile(r"\\[(.+?)\\]", re.DOTALL)

def extract_equations(text: str) -> list[str]:
    eqs = MATH_INLINE.findall(text) + MATH_BLOCK.findall(text)
    return [e.strip() for e in eqs]

def latex_nat_equality_to_lean(eq: str) -> Optional[str]:
    """
    Handle simple a+b=b+a or equalities of numerals only; map to Nat.
    """
    s = eq.replace(" ", "")
    if "=" not in s:
        return None
    lhs, rhs = s.split("=",1)
    # numerals + '+' only?
    ok = all(ch.isdigit() or ch in "+()" for ch in lhs+rhs)
    if ok:
        return f"theorem numeral_equality : ({lhs}) = ({rhs}) := by decide"
    # commutativity pattern a+b=b+a (symbolic a,b -> treat as Nat variables)
    # crude detection:
    if "+" in lhs and "+" in rhs:
        a,b = lhs.split("+",1)
        c,d = rhs.split("+",1)
        # check if rhs variables swapped
        if a==d and b==c:
            return "theorem add_comm_inst : a + b = b + a := by simpa using Nat.add_comm a b"
    return None
