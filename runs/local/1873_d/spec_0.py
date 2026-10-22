APPLICABLE = True
from collections import deque
from hypothesis import strategies as st

@st.composite
def gen(draw):
    t = draw(st.integers(1, 4))
    lines = [str(t)]
    for _ in range(t):
        n = draw(st.integers(1, 8))
        k = draw(st.integers(1, n))
        s = draw(st.text(alphabet="BW", min_size=n, max_size=n))
        lines.append(f"{n} {k}")
        lines.append(s)
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
    pos = 1
    total = 0
    for _ in range(t):
        if pos + 2 > len(tok):
            return False
        try:
            n = int(tok[pos]); k = int(tok[pos + 1])
        except ValueError:
            return False
        s = tok[pos + 2] if pos + 2 < len(tok) else None
        pos += 3
        if s is None:
            return False
        if not (1 <= k <= n <= 2 * 10**5):
            return False
        if len(s) != n or any(c not in "BW" for c in s):
            return False
        total += n
    if total > 2 * 10**5:
        return False
    return pos == len(tok)

def ref(inp):
    tok = inp.split()
    t = int(tok[0])
    pos = 1
    out = []
    for _ in range(t):
        n = int(tok[pos]); k = int(tok[pos + 1]); s = tok[pos + 2]
        pos += 3
        start = 0
        for i, c in enumerate(s):
            if c == 'B':
                start |= (1 << i)
        ops = []
        for i in range(n - k + 1):
            ops.append(((1 << k) - 1) << i)
        dist = {start: 0}
        dq = deque([start])
        while dq:
            cur = dq.popleft()
            if cur == 0:
                break
            for m in ops:
                nxt = cur & ~m
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    dq.append(nxt)
        out.append(str(dist[0]))
    return "\n".join(out)
