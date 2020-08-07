from typing import List, Tuple, NamedTuple

INF = 1e10


class Target(NamedTuple):
    pos: int
    val: int


def prefix_max(xs: List[int]) -> List[int]:
    ys = [xs[0]]
    for x in xs[1:]:
        ys.append(max(ys[-1], x))
    return ys


def solution(targets: List[Tuple[int, int]]) -> int:
    n = len(targets)
    t = [Target(targets[0][0], 0)] + [Target(*xs) for xs in targets]
    t.sort()
    f = [0] + [INF] * n  # f[i]: best plan with i being a "right supporting point"

    for i in range(n):
        # payload = t[l].val + (x - t[l].pos) == t[r].val + (t[r].pos - x)
        # x = (t[r].val + t[r].pos - t[l].val + t[l].pos) / 2
        # payload = t[r].val + (t[r].pos - (t[r].val + t[r].pos - t[l].val + t[l].pos) / 2)
        #         = t[r].val + t[r].pos - t[r].val/2 - t[r].pos/2 + t[l].val/2 - t[l].pos/2
        #         = (t[r].val + t[r].pos + t[l].val - t[l].pos) / 2
        vals = []
        for k in range(i + 1, n + 1):
            strength = max(0, t[k].val - max(0, t[i].val - (t[k].pos - t[i].pos)))
            vals.append(strength - t[k].pos if strength > 0 else -INF)
        left = prefix_max(vals)
        for j in range(i + 1, n + 1):
            min_payload = INF
            right = t[j].pos + t[j].val
            for k in range(j, i, -1):
                if t[k].pos + t[k].val > right: break
                l, r = left[k - (i + 1)], right
                if l == -INF: break
                payload = (l + r) / 2
                position = t[j].val + t[j].pos - payload
                if position >= t[k - 1].pos:
                    min_payload = min(min_payload, payload)
            f[j] = min(f[j], f[i] + min_payload)

    print(f)

    return int(f[-1])


EPS = 1e-4


def solution_wrong(targets: List[Tuple[int, int]]) -> int:
    targets = [Target(*xs) for xs in targets]
    targets.sort()
    f = [0]

    def check(payload: float, ts: List[Target]) -> bool:
        left, right = 0, INF
        for target in ts:
            radius = payload - target.val
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

    with flutes.work_in_progress():
        print(solution(targets))
    with flutes.work_in_progress():
        print(solution_wrong(targets))


if __name__ == '__main__':
    main()
