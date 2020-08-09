from collections import defaultdict
from typing import Dict, List, Tuple


def gaussian_elimination(mat: List[List[float]]) -> List[float]:
    n, m = len(mat), len(mat[0])
    assert n < m
    for i in range(n - 1):
        _, j = max((abs(mat[j][i]), j) for j in range(i, n))
        mat[i], mat[j] = mat[j], mat[i]
        for j in range(i + 1, n):
            # if abs(mat[j][i]) < EPS: continue
            d = mat[j][i] / mat[i][i]
            for k in range(i, m):
                mat[j][k] -= d * mat[i][k]
    sol = [0.] * n
    for i in range(n - 1, -1, -1):
        x = mat[i][-1]
        for j in range(i + 1, n):
            x -= mat[i][j] * sol[j]
        x /= mat[i][i]
        sol[i] = x
    return sol


def to_dense(d: Dict[int, float], length: int) -> List[float]:
    arr = [0.] * length
    for k, v in d.items():
        arr[k] = float(v)
    return arr


# @profile
def solution(n: int, snakes: List[Tuple[int, int]], ladders: List[Tuple[int, int]]) -> int:
    move = list(range(n))
    for a, b in snakes + ladders:
        move[a] = b
    snake_idx = [-1] * n
    for idx, (_, b) in enumerate(snakes):
        snake_idx[b] = idx
    m = len(snakes)

    coefs = [None] * n
    key_indices = sorted([0, n - 2] + [a for a, b in ladders + snakes] + [b for a, b in ladders + snakes])
    max_diff = max(b - a for a, b in zip(key_indices, key_indices[1:]))
    indices = set()
    for i in key_indices:
        for j in range(max(0, i - 5), min(n - 1, i + 6)):
            indices.add(j)
    indices = sorted(indices, reverse=True)

    trans_coefs = [[0.] * 7 for _ in range(6)]
    trans_coefs[0][0] = 1.
    for i in range(1, 6):
        trans_coefs[i][i + 1] = 1.
    trans_coefs.append([1.] + [1. / 6] * 6)
    for i in range(max_diff):
        next_coefs = [sum(trans_coefs[k][j] for k in range(i + 1, i + 7)) / 6 for j in range(7)]
        next_coefs[0] += 1.
        trans_coefs.append(next_coefs)
    trans_coefs = trans_coefs[5:]

    last_key = n - 1
    for i in indices:
        coefs[i] = cur_coef = defaultdict(float)
        trans = trans_coefs[last_key - i]
        cur_coef[m] += trans_coefs[last_key - i][0]
        for j in range(6):
            if last_key + j < n - 1:
                x = move[last_key + j]
                if snake_idx[x] != -1:
                    cur_coef[snake_idx[x]] += trans[6 - j]
                else:
                    for k, v in coefs[x].items():
                        cur_coef[k] += v * trans[6 - j]
        if all(i + j >= n - 1 or coefs[i + j] is not None for j in range(6)):
            last_key = i
    # print(coefs)

    mat = [to_dense(coefs[b], m + 1) for _, b in snakes]
    # mat = [coefs[b] for _, b in snakes]
    for i in range(len(snakes)):
        mat[i][i] -= 1.0
        mat[i][-1] = -mat[i][-1]
    # print([row[0] for row in mat])
    # print([row[-1] for row in mat])
    sol = gaussian_elimination(mat) + [1.]
    ans = sum(val * sol[pos] for pos, val in coefs[0].items())
    # ans = sum(val * sol[idx] for idx, val in enumerate(coefs[0]))
    print(ans)
    return int(ans)


def solution_unoptimized(n: int, snakes: List[Tuple[int, int]], ladders: List[Tuple[int, int]]) -> int:
    move = list(range(n))
    for a, b in snakes + ladders:
        move[a] = b
    snake_idx = [-1] * n
    for idx, (_, b) in enumerate(snakes):
        snake_idx[b] = idx

    coef = [[0.] * (len(snakes) + 1) for _ in range(n - 1)]
    one_sixth = 1. / 6
    for i in range(n - 2, -1, -1):
        coef[i][-1] = 1.
        for j in range(1, 7):
            if i + j < n - 1:
                x = move[i + j]
                if snake_idx[x] != -1:
                    coef[i][snake_idx[x]] += one_sixth
                else:
                    for k in range(len(snakes) + 1):
                        coef[i][k] += coef[x][k] * one_sixth
    # print(coef)

    # print([row[-1] for row in coef])
    mat = [coef[b] for _, b in snakes]
    for i in range(len(snakes)):
        mat[i][i] -= 1.0
        mat[i][-1] = -mat[i][-1]
    print([row[0] for row in mat])
    print([row[-1] for row in mat])
    sol = gaussian_elimination(mat)
    ans = coef[0][-1]
    for i in range(len(snakes)):
        ans += coef[0][i] * sol[i]
    print(ans)
    return int(ans)


def solution_brute(n: int, snakes: List[Tuple[int, int]], ladders: List[Tuple[int, int]]) -> int:
    move = list(range(n))
    for a, b in snakes + ladders:
        move[a] = b

    mat = [[0.] * (n + 1) for _ in range(n - 1)]
    for i in range(n - 1):
        mat[i][i] = mat[i][-1] = 1.0
        for j in range(1, 7):
            if i + j < n:
                mat[i][move[i + j]] += -1 / 6

    sol = gaussian_elimination(mat)
    print(sol[0])
    return int(sol[0])


EPS = 1e-8


def solution_iter(n: int, snakes: List[Tuple[int, int]], ladders: List[Tuple[int, int]]) -> int:
    exp = [0.] * n
    move = list(range(n))
    for a, b in snakes + ladders:
        move[a] = b
    prev = 0.
    for _ in range(1000000):
        for x in range(n - 2, -1, -1):
            expected = 0.
            for i in range(1, 7):
                if x + i < n:
                    expected += exp[move[x + i]]
            exp[x] = expected / 6 + 1
        # print(exp[0])
        if abs(exp[0] - prev) < EPS:
            break
        prev = exp[0]

    # print(exp)
    return int(exp[0])


import pickle
import random
import flutes


def gen_data(n: int, n_snakes: int, n_ladders: int):
    snakes = []
    ladders = []
    chosen = set()
    for _ in range(n_snakes + n_ladders):
        while True:
            a, b = random.randint(1, n - 2), random.randint(1, n - 2)
            if a != b and a not in chosen and b not in chosen: break
        chosen.add(a)
        chosen.add(b)
        a, b = min(a, b), max(a, b)
        if _ < n_snakes:
            snakes.append((b, a))
        else:
            ladders.append((a, b))
    return n, snakes, ladders


def main():
    random.seed(flutes.__MAGIC__)
    with open("data/sl_snakes_and_ladders.pkl", "rb") as f:
        n, snakes, ladders = pickle.load(f)

    n, snakes, ladders = gen_data(100000, 100, 100)
    with flutes.work_in_progress():
        print(solution(n, snakes, ladders))
    with flutes.work_in_progress():
        print(solution_unoptimized(n, snakes, ladders))

    n = 10000
    snakes = []
    ladders = []
    for i in range(80):
        snakes.append((n - 200 + i + (i // 5), i + 1 + (i // 5)))

    with flutes.work_in_progress():
        print(solution(n, snakes, ladders))
    with flutes.work_in_progress():
        print(solution_unoptimized(n, snakes, ladders))
    # with flutes.work_in_progress():
    #     print(solution_brute(n, snakes, ladders))
    # with flutes.work_in_progress():
    #     print(solution_iter(n, snakes, ladders))

    return

    cases = 0
    while True:
        n, snakes, ladders = gen_data(100000, 100, 100)
        out = solution(n, snakes, ladders)
        ans = solution_unoptimized(n, snakes, ladders)
        if out != ans:
            breakpoint()
        else:
            cases += 1
            print(f"Passed case {cases}")


if __name__ == '__main__':
    main()
