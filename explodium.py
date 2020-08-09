from typing import List, Tuple, NamedTuple

INF = 1e20


class Target(NamedTuple):
    pos: int
    val: int


def solution(targets: List[Tuple[int, int]]) -> int:
    n = len(targets)
    targets = sorted([Target(*xs) for xs in targets])
    ts = []
    for t in targets:
        while len(ts) > 0 and (t.val - ts[-1].val) >= (t.pos - ts[-1].pos):
            ts.pop(-1)
        if len(ts) > 0 and (ts[-1].val - t.val) >= (t.pos - ts[-1].pos):
            continue
        ts.append(t)
    n = len(ts)

    f = [0]
    min_val = INF
    for i in range(0, n):
        min_val = min(min_val, f[-1] + (ts[i].val - ts[i].pos) / 2)
        f.append(min_val + (ts[i].val + ts[i].pos) / 2)

    # print(f)
    return int(f[-1])


def prefix_max(xs: List[int]) -> List[int]:
    ys = [xs[0]]
    for x in xs[1:]:
        ys.append(max(ys[-1], x))
    return ys


def suffix_max(xs: List[int]) -> List[int]:
    return prefix_max(xs[::-1])[::-1]


def solution_brute(targets: List[Tuple[int, int]]) -> int:
    n = len(targets)
    ts = [Target(*xs) for xs in targets]
    ts.sort()
    f = [0]

    for i in range(n):
        val = INF
        for j in range(i, -1, -1):
            left = prefix_max([ts[k].val - ts[k].pos for k in range(j, i + 1)])
            right = suffix_max([ts[k].val + ts[k].pos for k in range(j, i + 1)])
            min_payload = INF
            for k in range(i, j - 1, -1):
                l, r = left[k - j], right[k - j]
                payload = (l + r) / 2
                position = r - payload
                if ts[k].pos <= position <= ts[min(k + 1, i)].pos:
                    min_payload = min(min_payload, payload)
            val = min(val, f[j] + min_payload)
        f.append(val)

    print(f)
    return int(f[-1])


import random
import pickle
import flutes


def main():
    random.seed(flutes.__MAGIC__)
    targets = [(0, 2), (2, 3), (8, 3)]
    with open("data/sl_explodium.pkl", "rb") as f:
        targets, = pickle.load(f)
    # targets = [(0, 8), (2, 8), (7, 4), (8, 4), (10, 2)]

    with flutes.work_in_progress():
        print(solution(targets))
    with flutes.work_in_progress():
        print(solution_brute(targets))


if __name__ == '__main__':
    main()
