from typing import Tuple, List

MAX_CANDIDATES = 2


def solution(s: str) -> int:
    is_prime = [False, False]
    for x in range(2, 10000):
        prime = True
        for y in range(2, min(x, 100)):
            if x % y == 0:
                prime = False
                break
            if y * y > x: break
        is_prime.append(prime)

    # @profile
    def dfs(s: str, candidates, max_num: int, max_digits: int) -> int:
        for digits in range(max_digits - 1, 0, -1):
            if len(candidates) >= MAX_CANDIDATES: break
            for l in range(len(s) - digits + 1):
                if s[l] == '0': continue
                num = int(s[l:(l + digits)])
                if is_prime[num] and num < max_num:
                    candidates.append((num, l, digits))
        candidates.sort(reverse=True)

        ret = 0
        for num, l, digits in candidates[:MAX_CANDIDATES]:
            next_s = s[:l] + s[(l + digits):]
            next_candidates = []
            for xs in candidates:
                if xs[0] >= num: continue
                if xs[1] + xs[2] <= l:
                    next_candidates.append(xs)
                elif xs[1] >= l + digits:
                    next_candidates.append((xs[0], xs[1] - digits, xs[2]))
            for next_l in range(max(0, l - digits + 1), min(l, len(next_s) - digits + 1)):
                if next_s[next_l] == '0': continue
                next_num = int(next_s[next_l:(next_l + digits)])
                if is_prime[next_num] and next_num < num:
                    next_candidates.append((next_num, next_l, digits))
            ret = max(ret, num + dfs(next_s, next_candidates, num, digits))
        return ret

    ans = dfs(s, [], 10000, 5)
    return ans


INVALID = -12345678


def solution_wrong(s: str) -> int:
    is_prime = [False, False]
    for x in range(2, 10000):
        prime = True
        for y in range(2, min(x, 100)):
            if x % y == 0:
                prime = False
                break
            if y * y > x: break
        is_prime.append(prime)

    f = [[INVALID if l <= r else 0 for r in range(len(s))] for l in range(len(s))]
    for l in range(len(s)):
        if is_prime[int(s[l])]:
            f[l][l] = int(s[l])
    for length in range(2, len(s) + 1):
        for l in range(len(s) - length + 1):
            r = l + length - 1
            val = INVALID
            for i in range(l, r):
                val = max(val, f[l][i] + f[i + 1][r])
            if s[l] == '0': continue
            two = int(s[l] + s[r])
            if is_prime[two]:
                val = max(val, two + f[l + 1][r - 1])
            for i in range(l + 1, r):
                three = int(s[l] + s[i] + s[r])
                if is_prime[three]:
                    val = max(val, three + f[l + 1][i - 1] + f[i + 1][r - 1])
                    if l == 1 and r == 5:
                        print(val, three, is_prime[three], f[l + 1][i - 1], f[i + 1][r - 1])
                for j in range(i + 1, r):
                    four = int(s[l] + s[i] + s[j] + s[r])
                    if is_prime[four]:
                        val = max(val, four + f[l + 1][i - 1] + f[i + 1][j - 1] + f[j + 1][r - 1])
            f[l][r] = val
    ans = max(0, f[0][len(s) - 1])
    return ans


import random
import flutes


def main():
    random.seed(flutes.__MAGIC__)
    s = "297891"

    n = 75
    s = "".join(chr(48 + random.randint(0, 9)) for _ in range(n))

    s = "000001"

    with flutes.work_in_progress():
        print(solution(s))


if __name__ == '__main__':
    main()
