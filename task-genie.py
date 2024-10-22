from typing import List, Tuple

INF = 10 ** 9


def solution_n2(tasks: List[Tuple[int, int]], wishes: int) -> int:
    n = len(tasks)
    children = [[] for _ in range(n)]
    cost = [c for _, c in tasks]
    for idx, (p, _) in enumerate(tasks):
        if p != -1: children[p].append(idx)
    size = [1] * n
    max_depth = [0] * n

    def dfs(x: int) -> None:
        for child in children[x]:
            dfs(child)
            size[x] += size[child]
            max_depth[x] = max(max_depth[x], max_depth[child])
        max_depth[x] += cost[x]

    dfs(0)

    def dp(x: int) -> List[int]:
        if size[x] == 1:
            return [cost[x], 0]
        f = dp(children[x][0])
        for child in children[x][1:]:
            g = dp(child)
            next_f = []
            next_f = [INF] * min(wishes + 1, len(f) + len(g) - 1)
            for i, f_val in enumerate(f):
                for j, g_val in enumerate(g):
                    if i + j >= len(next_f): break
                    next_f[i + j] = min(next_f[i + j], max(f_val, g_val))
            f = next_f
        return [max_depth[x]] + [min(f[i], f[i + 1] + cost[x]) for i in range(len(f) - 1)]

    f = dp(0)
    ans = f[wishes]
    return ans


def solution_n3(tasks: List[Tuple[int, int]], wishes: int) -> int:
    n = len(tasks)
    children = [[] for _ in range(n)]
    cost = [c for _, c in tasks]
    for idx, (p, _) in enumerate(tasks):
        if p != -1: children[p].append(idx)
    size = [1] * n
    max_depth = [0] * n

    def dfs(x: int) -> None:
        for child in children[x]:
            dfs(child)
            size[x] += size[child]
            max_depth[x] = max(max_depth[x], max_depth[child])
        max_depth[x] += cost[x]

    dfs(0)

    def dp(x: int) -> List[int]:
        if size[x] == 1:
            return [cost[x], 0]
        f = dp(children[x][0])
        for child in children[x][1:]:
            g = dp(child)
            next_f = [INF] * min(wishes + 1, len(f) + len(g) - 1)
            for i, f_val in enumerate(f):
                for j, g_val in enumerate(g):
                    if i + j >= len(next_f): break
                    next_f[i + j] = min(next_f[i + j], max(f_val, g_val))
            f = next_f
        return [max_depth[x]] + [min(f[i], f[i + 1] + cost[x]) for i in range(len(f) - 1)]

    f = dp(0)
    ans = f[wishes]
    return ans


def solution_wrong(tasks: List[Tuple[int, int]], wishes: int) -> int:
    n = len(tasks)
    children = [[] for _ in range(n)]
    parent = [p for p, _ in tasks]
    costs = [c for _, c in tasks]
    for idx, p in enumerate(parent):
        if p != -1: children[p].append(idx)
    depth = [0] * n
    chain_sum = [0] * n
    max_len = [0] * n
    subtree = [0] * n

    def find_depth(x: int) -> int:
        depth[x] = max((find_depth(ch) for ch in children[x]), default=0) + 1
        return depth[x]

    def find_longest_chain(x: int) -> int:
        if x != 0: chain_sum[x] = chain_sum[parent[x]] + costs[parent[x]]
        max_len[x] = max((find_longest_chain(ch) for ch in children[x]), default=0) + costs[x]
        return max_len[x]

    def find_chains_in_subtree(x: int) -> int:
        if len(children[x]) == 0:
            subtree[x] = int(max_len[x] + chain_sum[x] == max_len[0])
        else:
            subtree[x] = sum((find_chains_in_subtree(ch) for ch in children[x]), 0)
        return subtree[x]

    find_depth(0)
    for _ in range(wishes):
        find_longest_chain(0)
        find_chains_in_subtree(0)
        *_, idx = max((subtree[x], costs[x], x) for x in range(n) if subtree[x] == subtree[0] and costs[x] > 0)
        costs[idx] = 0
    return find_longest_chain(0)


import pickle
import flutes


def main():
    with open("data/sl_task_genie.pkl", "rb") as f:
        tasks, wishes = pickle.load(f)
    with flutes.work_in_progress():
        print(solution_n2(tasks, wishes))
    with flutes.work_in_progress():
        print(solution_n3(tasks, wishes))
    with flutes.work_in_progress():
        print(solution_wrong(tasks, wishes))


if __name__ == '__main__':
    main()
