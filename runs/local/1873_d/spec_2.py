APPLICABLE = True
from hypothesis import strategies as st
from collections import deque


@st.composite
def gen(draw):
    t = draw(st.integers(1, 4))
    lines = [str(t)]
    for _ in range(t):
        n = draw(st.integers(1, 10))
        k = draw(st.integers(1, n))
        s = draw(st.text(alphabet="BW", min_size=n, max_size=n))
        lines.append(f"{n} {k}")
        lines.append(s)
    return "\n".join(lines) + "\n"


def validate(inp):
    try:
        tok = inp.split()
        if len(tok) < 1:
            return False
        t = int(tok[0])
        if not (1 <= t <= 1000):
            return False
        if len(tok) != 1 + 3 * t:
            return False
        total = 0
        idx = 1
        for _ in range(t):
            n = int(tok[idx])
            k = int(tok[idx + 1])
            s = tok[idx + 2]
            idx += 3
            if not (1 <= k <= n <= 2 * 10**5):
                return False
            if len(s) != n or any(c not in "BW" for c in s):
                return False
            total += n
        return total <= 2 * 10**5
    except Exception:
        return False


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
        # Bitmask of black cells; cell i corresponds to bit i.
        start = 0
        for i, c in enumerate(s):
            if c == 'B':
                start |= (1 << i)
        # Windows of k consecutive cells, as bitmasks.
        windows = [((1 << k) - 1) << i for i in range(n - k + 1)]
        # BFS over states: minimum number of operations to reach 0 black cells.
        dist = {start: 0}
        q = deque([start])
        ans = None
        while q:
            cur = q.popleft()
            if cur == 0:
                ans = dist[cur]
                break
            for w in windows:
                nxt = cur & ~w
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        out.append(str(ans))
    return "\n".join(out)
