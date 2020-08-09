from typing import List, Tuple


def solution_brute(tasks: List[Tuple[int, int]], wishes: int) -> int:
    n = len(tasks)
    children = [[] for _ in range(n)]
    for idx, (p, _) in enumerate(tasks):
        if p != -1: children[p].append(idx)

    def dfs(x: int) -> None:
        pass

    dfs(0)


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
        print(solution_wrong(tasks, wishes))


if __name__ == '__main__':
    main()
