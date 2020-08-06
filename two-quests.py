from typing import List

INF = 2000000000


# @profile
def solution(a: List[int], b: List[int]) -> int:
    n = len(a)
    m = len(b)
    a = [0] + a
    b = [0] + b
    diff_a = [0] + [abs(y - x) for x, y in zip(a, a[1:])] + [0] * (m + 1)
    diff_b = [0] + [abs(y - x) for x, y in zip(b, b[1:])] + [0] * (n + 1)
    f = [INF] * (n + m + 1)
    g = [INF] * (n + m + 1)
    # f[i]/g[i] -> (i, diag - i)
    f[1], g[0] = a[1], b[1]
    for diag in range(2, n + m + 1):
        # if diag <= n:
        f[diag] = f[diag - 1] + diff_a[diag]
        l, r = max(1, diag - m), min(n, diag - 1)
        for i in range(r, l - 1, -1):
            j = diag - i
            g[i] = min(f[i] + abs(b[j] - a[i]), g[i] + diff_b[j])
            f[i] = min(f[i - 1] + diff_a[i], g[i - 1] + abs(a[i] - b[j]))
        # if diag <= m:
        g[0] += diff_b[diag]
    return min(f[n], g[n])


def solution_(a: List[int], b: List[int]) -> int:
    n = len(a)
    m = len(b)
    a = [0] + a
    b = [0] + b
    diff_a = [0] + [abs(y - x) for x, y in zip(a, a[1:])] + [0] * (m + 1)
    diff_b = [0] + [abs(y - x) for x, y in zip(b, b[1:])] + [0] * (n + 1)
    f = [INF] * (n + m + 1)
    g = [INF] * (n + m + 1)
    f2 = [INF] * (n + m + 1)
    g2 = [INF] * (n + m + 1)
    # f[i]/g[i] -> (i, diag - i)
    f[1], g[0] = a[1], b[1]
    # print()
    # print(_repr(f), _repr(g))
    for diag in range(2, n + m + 1):
        # if diag <= m:
        g2[0] = g[0] + diff_b[diag]
        # if diag <= n:
        f2[diag] = f[diag - 1] + diff_a[diag]
        for i in range(max(1, diag - m), min(n, diag - 1) + 1):
            j = diag - i
            f2[i] = min(f[i - 1] + diff_a[i], g[i - 1] + abs(a[i] - b[j]))
            g2[i] = min(f[i] + abs(b[j] - a[i]), g[i] + diff_b[j])
        f, f2 = f2, f
        g, g2 = g2, g
        # print(_repr(f), _repr(g), sep='    ')
    return min(f[n], g[n])


def _repr(s):
    return repr(s).replace(str(INF), 'inf')


class Array2D:
    def __init__(self, n: int, m: int, default_val: int = 0):
        self.n = n
        self.m = m
        # self.bits = 0
        # while (1 << self.bits) < m:
        #     self.bits += 1
        # self.arr = [default_val] * (n * (1 << self.bits))
        self.arr = [default_val] * (n * m)

    def __getitem__(self, item):
        # return self.arr[(item[0] << self.bits) + item[1]]
        return self.arr[(item[0] * self.m) + item[1]]

    def __setitem__(self, key, value):
        # self.arr[(key[0] << self.bits) + key[1]] = value
        self.arr[(key[0] * self.m) + key[1]] = value

    def __repr__(self) -> str:
        lines = [f"<Array2D {self.n} x {self.m}>"]
        for r in range(self.n):
            row = self.arr[(r * self.m):((r + 1) * self.m)]
            lines.append(_repr(row))
        return "\n".join(lines)


def solution_full(a: List[int], b: List[int]) -> int:
    n = len(a)
    m = len(b)
    a = [0] + a
    b = [0] + b
    f = Array2D(n + 1, m + 1, default_val=INF)
    g = Array2D(n + 1, m + 1, default_val=INF)
    f[0, 0] = g[0, 0] = 0
    for i in range(1, n + 1):
        f[i, 0] = f[i - 1, 0] + abs(a[i] - a[i - 1])
    for j in range(1, m + 1):
        g[0, j] = g[0, j - 1] + abs(b[j] - b[j - 1])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            f[i, j] = min(f[i - 1, j] + abs(a[i] - a[i - 1]), g[i - 1, j] + abs(a[i] - b[j]))
            g[i, j] = min(f[i, j - 1] + abs(b[j] - a[i]), g[i, j - 1] + abs(b[j] - b[j - 1]))
    # print(f)
    # print(g)
    return min(f[n, m], g[n, m])


import itertools


def solution_brute(a: List[int], b: List[int]) -> int:
    ans = INF
    for plan in itertools.combinations(range(1, len(a) + len(b) + 1), len(a)):
        seq = [-1] * (len(a) + len(b) + 1)
        seq[0] = 0
        for i, x in zip(plan, a):
            seq[i] = x
        idx = 0
        for i in range(len(seq)):
            if seq[i] == -1:
                seq[i] = b[idx]
                idx += 1
        val = sum(abs(y - x) for x, y in zip(seq, seq[1:]))
        ans = min(ans, val)
        # if val == 270199:
        #     print(plan)
    return ans


import random
import flutes


def main():
    random.seed(flutes.__MAGIC__)
    a = [5, 3, 10, 6]
    b = [9, 7, 12]

    n = 1000
    m = 1000
    a = [random.randint(0, 99999) for _ in range(n)]
    b = [random.randint(0, 99999) for _ in range(m)]

    # print(a)
    # print(b)
    with flutes.work_in_progress():
        print(solution(a, b))
    with flutes.work_in_progress():
        print(solution_(a, b))
    with flutes.work_in_progress():
        print(solution_full(a, b))
    with flutes.work_in_progress():
        print(solution_brute(a, b))


if __name__ == '__main__':
    main()
