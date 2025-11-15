from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict

def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)

def apply_1q(U: np.ndarray, psi: np.ndarray, n: int, q: int) -> np.ndarray:
    ops = [I]*n
    ops[n-1-q] = U  # little-endian / adjust if you prefer
    Ufull = kron_all(ops)
    return Ufull @ psi

def apply_cx(psi: np.ndarray, n: int, ctrl: int, targ: int) -> np.ndarray:
    dim = 2**n
    psi = psi.copy()
    for i in range(dim):
        if ((i >> ctrl) & 1) == 1 and ((i >> targ) & 1) == 0:
            j = i | (1 << targ)
            psi[i], psi[j] = psi[j], psi[i]
    return psi

def measure_expectation(psi: np.ndarray, n: int, op_on: Dict[int, np.ndarray]) -> float:
    ops = [I]*n
    for q, O in op_on.items():
        ops[n-1-q] = O
    O = kron_all(ops)
    return float(np.real(np.vdot(psi, O @ psi)))

def run_ghz(n: int=3) -> Dict[str, float]:
    psi = np.zeros(2**n, dtype=complex); psi[0] = 1.0
    psi = apply_1q(H, psi, n, q=0)
    for k in range(1, n):
        psi = apply_cx(psi, n, ctrl=0, targ=k)
    exps = {
        "Z_all": measure_expectation(psi, n, {q: Z for q in range(n)}),
        "X_all": measure_expectation(psi, n, {q: X for q in range(n)}),
    }
    return exps
