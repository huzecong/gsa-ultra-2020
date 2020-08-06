from typing import List, Tuple


class RMQ:
    def __init__(self, arr: List[int]):
        rmq: List[List[int]] = [arr]
        cur_len = 1
        log = [0] * (len(arr) + 1)
        while cur_len * 2 <= len(arr):
            prev = rmq[-1]
            for i in range(cur_len * 2 + 1, min(len(log), cur_len * 4 + 1)):
                log[i] = len(rmq)
            vals = [max(prev[i], prev[i + cur_len]) for i in range(len(prev) - cur_len)]
            rmq.append(vals)
            cur_len *= 2
        self.log = log
        self.rmq = rmq

    def query(self, l: int, r: int) -> int:
        if l == r: return 0
        idx = self.log[r - l]
        return max(self.rmq[idx][l], self.rmq[idx][r - (1 << idx)])


def solution(arr: List[int], qs: List[Tuple[int, int]]) -> int:
    # block_size = max(1, int(math.log2(len(arr)) / 4))
    block_size = 16
    maxima = [max(arr[idx:(idx + block_size)]) for idx in range(0, len(arr), block_size)]
    rmq = RMQ(maxima)
    ans = 0
    for l, r in qs:
        lb, rb = l >> 4, r >> 4
        if lb == rb:
            val = max(arr[l:r])
        else:
            val = 0
            if l & 15 > 0:
                val = max(arr[l:((lb + 1) << 4)])
                lb += 1
            if r & 15 > 0:
                val = max(val, max(arr[(rb << 4):r]))
            val = max(val, rmq.query(lb, rb))
        ans += val
    return ans


# @profile
def solution_rmq(arr: List[int], qs: List[Tuple[int, int]]) -> int:
    rmq = RMQ(arr)
    ans = 0
    for l, r in qs:
        ans += rmq.query(l, r)
    return ans


class BIT:
    def __init__(self, n: int):
        self.n = n
        self.arr = [0] * (n + 1)

    def query(self, x: int) -> int:
        r = 0
        while x > 0:
            r = max(r, self.arr[x])
            x -= x & -x
        return r

    def modify(self, x: int, v: int) -> None:
        while x <= self.n:
            self.arr[x] = max(self.arr[x], v)
            x += x & -x


def solution2(arr: List[int], qs: List[Tuple[int, int]]) -> int:
    t = BIT(len(arr))
    qs.sort(reverse=True)
    prev = len(arr)
    ans = 0
    for l, r in qs:
        while prev > l:
            t.modify(prev, arr[prev - 1])
            prev -= 1
        ans += t.query(r)
    return ans


def solution_brute(arr: List[int], qs: List[Tuple[int, int]]) -> int:
    return sum(max(arr[l:r]) for l, r in qs)


import random

import flutes


def main():
    random.seed(flutes.__MAGIC__)
    a = [3, 6, 6, 1, 9, 8, 4, 7, 1, 1]
    qs = [(1, 7), (5, 7), (0, 1)]
    n = 100000
    q = 100000
    a = [random.randint(1, 99999) for _ in range(n)]
    qs = []
    for _ in range(q):
        while True:
            l = random.randint(0, n - 1)
            r = random.randint(0, n - 1)
            if l != r: break
        l, r = min(l, r), max(l, r)
        qs.append((l, r))
    with flutes.work_in_progress():
        print(solution(a, qs))
    with flutes.work_in_progress():
        print(solution_rmq(a, qs))
    with flutes.work_in_progress():
        print(solution2(a, qs))
    with flutes.work_in_progress():
        print(solution_brute(a, qs))


if __name__ == '__main__':
    main()
