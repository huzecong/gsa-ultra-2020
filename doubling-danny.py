import random

import flutes


def solution_truth(a, m):
    return sum(pow(2, x % (m - 1), m) for x in a) % m


def fastmul(a: int, b: int, m: int) -> int:
    r = 0
    while b > 0:
        if b & 1:
            r += a
            if r >= m: r -= m
        a <<= 1
        if a >= m: a -= m
        b >>= 1
    return r


# @profile
def _solution(a, mod, n_iters=5, prev_sum_log=None):
    orders = sorted((x, i) for i, x in enumerate(a))
    powers = [0] + [x for x, _ in orders]
    powers = [b - a for a, b in zip(powers, powers[1:])]
    binaries = [bin(x)[2:] for x in powers]
    sum_log = sum(len(x) for x in binaries)

    ans = [None] * len(a)
    if prev_sum_log is not None and prev_sum_log <= sum_log + len(a): return None
    res = None
    if n_iters > 0:
        res = _solution(powers, mod, n_iters - 1, sum_log)
    if res is None:
        pow2 = [2]
        max_log = max(len(x) for x in binaries)
        for _ in range(max_log):
            val = pow2[-1] ** 2 % mod
            pow2.append(val)
        prev = 1
        for idx, b in enumerate(binaries):
            x = 1
            for i in range(len(b)):
                if b[-(i + 1)] == '0': continue
                x *= pow2[i]
                if x >= mod: x %= mod
            x *= prev
            if x >= mod: x %= mod
            ans[orders[idx][1]] = prev = x
    else:
        x = 1
        for idx, v in enumerate(res):
            x *= v
            if x >= mod: x %= mod
            ans[orders[idx][1]] = x
    return ans


def solution(a, m):
    a = [x % (m - 1) for x in a]
    ret = 0
    for x in _solution(a, m):
        ret += x
        if ret >= m: ret -= m
    return ret


def solution_(a, m):
    powers = [0] + sorted(x % (m - 1) for x in a)
    powers = [b - a for a, b in zip(powers, powers[1:])]

    binaries = [bin(x)[2:] for x in powers]
    max_log = max(len(x) for x in binaries)
    pow2 = [2]
    for _ in range(max_log):
        val = pow2[-1] ** 2
        if val >= m: val %= m
        pow2.append(val)
    prev = 1
    ans = 0
    for b in binaries:
        x = 1
        for i, v in zip(reversed(b), pow2):
            if i == '0': continue
            x *= v
            if x >= m: x %= m
        x *= prev
        if x >= m: x %= m
        prev = x
        ans += x
        if ans >= m: ans -= m
    return ans


def main():
    random.seed(19260817)
    # with open("/Users/kanari/Downloads/sl_doubling_danny.pkl", "rb") as f:
    #     data = pickle.load(f)
    # print(data)
    # print(solution(*data))
    with flutes.work_in_progress():
        m = 2 ** 2203 - 1
        x = 2 ** 2203
        # a = [m - random.randint(0, 10000) for _ in range(1000)]
        a = [random.randint(0, x) for _ in range(500)]
    with flutes.work_in_progress():
        print(solution(a, m))
    with flutes.work_in_progress():
        print(solution_(a, m))
    # with flutes.work_in_progress():
    #     print(solution_truth(a, m))


if __name__ == '__main__':
    main()
