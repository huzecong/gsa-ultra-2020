from typing import List, Tuple, NamedTuple

EPS = 1e-4
INF = 1e10


class Target(NamedTuple):
    pos: int
    strength: int


def solution(targets: List[Tuple[int, int]]) -> int:
    targets = [Target(*xs) for xs in targets]
    targets.sort()
    f = [0]  # f[i]: best plan with i being a "right supporting point"

    for i in range(len(targets)):



def solution_wrong(targets: List[Tuple[int, int]]) -> int:
    targets = [Target(*xs) for xs in targets]
    targets.sort()
    f = [0]

    def check(payload: float, ts: List[Target]) -> bool:
        left, right = 0, INF
        for target in ts:
            radius = payload - target.strength
            left = max(left, target.pos - radius)
            right = min(right, target.pos + radius)
            if left > right: return False
        return True

    for i in range(len(targets)):
        val = INF
        for j in range(i, -1, -1):
            ts = targets[j:(i + 1)]
            l = max(s for _, s in ts)
            r = l + (ts[-1].pos - ts[0].pos)
            while (r - l) > EPS:
                mid = (l + r) / 2
                if check(mid, ts):
                    r = mid
                else:
                    l = mid
            val = min(val, f[j] + l)
        f.append(val)
    # print(f[-1])
    return int(f[-1])


import random
import pickle
import flutes


def main():
    random.seed(flutes.__MAGIC__)
    targets = [(0, 2), (2, 3), (8, 3)]
    with open("data/sl_explodium.pkl", "rb") as f:
        targets, = pickle.load(f)

    with flutes.work_in_progress():
        print(solution(targets))
    with flutes.work_in_progress():
        print(solution_wrong(targets))


if __name__ == '__main__':
    main()
