APPLICABLE = True
from hypothesis import strategies as st
from collections import deque


@st.composite
def gen(draw):
    t = draw(st.integers(1, 4))
    lines = [str(t)]
    for _ in range(t):
        n = draw(st.integers(1, 8))
        k = draw(st.integers(1, n))
        s = draw(st.lists(st.sampled_from("BW"), min_size=n, max_size=n))
        lines.append(f"{n} {k}")
        lines.append("".join(s))
    return "\n".join(lines) + "\n"


def validate(inp):
    tok = inp.split()
    if not tok:
        return False
    try:
        t = int(tok[0])
    except ValueError:
        return False
    if not (1 <= t <= 1000):
        return False
    idx = 1
    total = 0
    for _ in range(t):
        if idx + 2 > len(tok):
            return False
        try:
            n = int(tok[idx])
            k = int(tok[idx + 1])
        except ValueError:
            return False
        s = tok[idx + 2]
        idx += 3
        if not (1 <= k <= n <= 2 * 10 ** 5):
            return False
        if len(s) != n or any(c not in "BW" for c in s):
            return False
        total += n
    if total > 2 * 10 ** 5:
        return False
    return idx == len(tok)


def _min_ops(n, k, s):
    start = 0
    for i, c in enumerate(s):
        if c == "B":
            start |= 1 << i
    windows = []
    for j in range(n - k + 1):
        w = 0
        for i in range(j, j + k):
            w |= 1 << i
        windows.append(w)
    dist = {start: 0}
    q = deque([start])
    while q:
        m = q.popleft()
        if m == 0:
            return dist[m]
        for w in windows:
            nm = m & ~w
            if nm not in dist:
                dist[nm] = dist[m] + 1
                q.append(nm)
    return -1


def ref(inp):
    tok = inp.split()
    t = int(tok[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(tok[idx])
        k = int(tok[idx + 1])
        s = tok[idx + 2]
        idx += 3
        out.append(str(_min_ops(n, k, s)))
    return "\n".join(out) + "\n"
