from typing import List

INF = 1e10

# TODO: Change to monotonic queue

@profile
def solution(ps: List[List[int]]) -> int:
    ps = [sorted(p) for p in ps]
    f = [[INF] * 10 for _ in ps[0]]
    for i in range(len(ps[0])):
        f[i][4] = 4

    for column, heights in enumerate(ps):
        if column == 0: continue
        prev_heights = ps[column - 1]
        ls, rs = [], []
        for i in range(len(prev_heights)):
            r = 10
            l = next((idx for idx, v in enumerate(f[i]) if v != INF), 10)
            if l < 10:
                r = next(idx for idx in range(9, -1, -1) if f[i][idx] != INF) + 1
            ls.append(l)
            rs.append(r)
        prev_idx = 0
        g = []
        for idx, h in enumerate(heights):
            while prev_idx < len(prev_heights) and prev_heights[prev_idx] < h - 5:
                prev_idx += 1
            cur_val = [INF] * 11
            for i in range(prev_idx, len(prev_heights)):
                ph = prev_heights[i]
                if ph > h + 5: break
                if ph < h:
                    for pace in range(ls[i], rs[i]):
                        x = min(9, pace + 1)
                        cur_val[x] = min(cur_val[x], f[i][pace])
                elif ph == h:
                    for pace in range(ls[i], rs[i]):
                        cur_val[pace] = min(cur_val[pace], f[i][pace])
                else:
                    for pace in range(ls[i], rs[i]):
                        x = max(0, pace - 1)
                        cur_val[x] = min(cur_val[x], f[i][pace])
            for pace in range(10):
                cur_val[pace] += pace
            g.append(cur_val)
        f = g
        # print([[v if v != INF else '' for v in fs] for fs in f])
    ans = min(min(fs) for fs in f) + len(ps)
    return ans


import flutes
import pickle


def main():
    with open("data/sl_hardcore_parkour.pkl", "rb") as f:
        ps, = pickle.load(f)
    n = 1000
    ps = [list(range(1, 20 + 1)) for _ in range(n)]
    # print(ps)
    with flutes.work_in_progress():
        print(solution(ps))


if __name__ == '__main__':
    main()
