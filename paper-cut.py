from typing import List, Tuple

MOD = 10 ** 9 + 7


def exgcd(a, b):
    # Base Case
    if a == 0:
        return b, 0, 1

    gcd, x1, y1 = exgcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def positive(x: int, y: int, a: int, b: int) -> bool:
    # Check if exists k s.t. x + k*b >= 0 and y + k*a >= 0
    if x >= 0 and y >= 0: return True
    if x < 0:
        x, y = y, x
        a, b = b, a
    d = (-y - 1) // a + 1
    assert y + d * a >= 0
    return x - d * b >= 0


def guess(a: int, b: int, c: int, d: int) -> bool:
    if a > b: a, b = b, a
    if c % a == 0 and d % b == 0: return True
    if c % b == 0 and d % a == 0: return True
    gcd, x, y = exgcd(a, b)
    lcm = a * b // gcd
    if c % lcm == 0 and d % gcd == 0 and positive(x * d, y * d, a, b): return True
    if d % lcm == 0 and c % gcd == 0 and positive(x * c, y * c, a, b): return True
    return False


def solution(qs: List[Tuple[int, int, int, int]]) -> int:
    ans = 0
    powtwo = 1
    for idx, (a, b, c, d) in enumerate(qs):
        if guess(a, b, c, d):
            ans += powtwo
            if ans >= MOD: ans -= MOD
        powtwo *= 2
        if powtwo >= MOD: powtwo -= MOD
    return ans


from functools import lru_cache


@lru_cache(maxsize=None)
def check(a: int, b: int, x: int, y: int) -> bool:
    if a > b: a, b = b, a
    if x > y: x, y = y, x
    if x == a and y == b:
        return True
    for i in range(a, x - a + 1):
        if check(a, b, i, y) and check(a, b, x - i, y):
            return True
    for i in range(a, y - a + 1):
        if check(a, b, x, i) and check(a, b, x, y - i):
            return True
    return False


def solution_brute(qs: List[Tuple[int, int, int, int]]) -> int:
    ans = 0
    for idx, (a, b, c, d) in enumerate(qs):
        if check(a, b, c, d):
            ans += pow(2, idx, MOD)
            if ans >= MOD: ans -= MOD
    return ans


import sys
import pickle
import flutes
import random
import tqdm
import matplotlib.pyplot as plt


def main():
    random.seed(flutes.__MAGIC__)
    sys.setrecursionlimit(32768)
    qs = [(3, 2, 5, 6)]

    with open("data/sl_paper_cut.pkl", "rb") as f:
        qs, = pickle.load(f)

    with flutes.work_in_progress():
        print(solution(qs))
    with flutes.work_in_progress():
        print(solution_brute(qs))

    # import numpy as np
    # arr = np.zeros((100, 100))
    # a, b = 5, 18
    # ok = []
    # for x in range(1, arr.shape[0]):
    #     for y in range(1, arr.shape[1]):
    #         if check(a, b, x, y):
    #             arr[x, y] = 1
    #             ok.append((x, y))
    # plt.imshow(arr)
    # plt.show()
    # print(ok)

    for _ in tqdm.trange(1000):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        c = random.randint(1, 100)
        d = random.randint(1, 100)
        ans = check(a, b, c, d)
        out = guess(a, b, c, d)
        if ans != out:
            print(f"{a, b, c, d}, expected: {ans}, received: {out}")
            guess(a, b, c, d)


if __name__ == '__main__':
    main()
