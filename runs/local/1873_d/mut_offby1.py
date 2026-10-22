import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    out = []
    for _ in range(t):
        n = int(data[idx]); k = int(data[idx + 1]); idx += 2
        s = data[idx]; idx += 1
        count = 0
        covered_until = 0
        for i in range(n):
            if s[i] == 'B' and i >= covered_until:
                count += 1
                covered_until = i + k - 1
        out.append(str(count))
    sys.stdout.write("\n".join(out) + "\n")

main()
