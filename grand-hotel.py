import heapq
from collections import defaultdict
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

T = TypeVar('T')

# profile = lambda x: x


class SegTree(Generic[T]):
    def __init__(self, n: int, default_val: T):
        self.n = n
        self.default_val = default_val
        self.nodes: Dict[int, int] = defaultdict(lambda: self.default_val)

    @profile
    def _query(self, x: int, l: int, r: int, s: int, t: int) -> T:
        if l == s and r == t: return self.nodes[x]
        mid = (l + r) >> 1
        if r <= mid: return self._query(x << 1, l, mid, s, t)
        if s > mid: return self._query(x << 1 | 1, mid + 1, r, s, t)
        return min(self._query(x << 1, l, mid, s, mid), self._query(x << 1 | 1, mid + 1, r, mid + 1, t))

    def query(self, l: int, r: int) -> T:
        return self._query(1, 1, self.n, l, r)

    @profile
    def modify(self, p: int, v: T) -> None:
        x, l, r = 1, 1, self.n
        while l != r:
            mid = (l + r) >> 1
            if p <= mid:
                x, r = x << 1, mid
            else:
                x, l = x << 1 | 1, mid + 1
        self.nodes[x] = v
        while x > 0:
            x >>= 1
            self.nodes[x] = min(self.nodes[x << 1], self.nodes[x << 1 | 1])


@profile
def solution(n: int, reservations: List[Tuple[str, int]]) -> int:
    INVALID = n + 10
    max_len = max((l for op, l in reservations if op == 'I'), default=1) + 1
    left: Dict[int, int] = {}  # (left endpoint) -> right
    right: Dict[int, int] = {}  # (right endpoint) -> left
    segments: Dict[int, List[int]] = defaultdict(list)  # (segment length) -> heap[left endpoint]
    segtree = SegTree[int](max_len, default_val=INVALID)
    assigned: List[Tuple[int, int]] = []  # (request id) -> (left, right endpoint)

    @profile
    def update(length: int, new_pos: Optional[int] = None):
        length = min(max_len, length)
        segs = segments[length]
        while len(segs) > 0 and segs[0] not in left:
            heapq.heappop(segs)
        if new_pos is not None:
            heapq.heappush(segs, new_pos)
        pos = segs[0] if len(segs) > 0 else INVALID
        segtree.modify(length, pos)

    def remove(l: int, r: int):
        del left[l]
        del right[r]
        update(r - l + 1)

    def add(l: int, r: int):
        left[l] = r
        right[r] = l
        update(r - l + 1, new_pos=l)

    add(0, n - 1)
    for op, _payload in reservations:
        if op == 'I':
            length = _payload
            l = segtree.query(length, max_len)
            r = left[l]
            assigned.append((l, l + length - 1))
            remove(l, r)
            if r - l + 1 > length:
                add(l + length, r)
        else:
            index = _payload
            l, r = assigned[index]
            assigned[index] = (-1, -1)
            if l - 1 in right:
                new_l = right[l - 1]
                remove(new_l, l - 1)
                l = new_l
            if r + 1 in left:
                new_r = left[r + 1]
                remove(r + 1, new_r)
                r = new_r
            add(l, r)

        # print(op, _payload, left, right, segments)
    ans = sum(idx * l for idx, (l, r) in enumerate(assigned) if l != -1)
    return ans


def solution_brute(n: int, rs: List[Tuple[str, int]]) -> int:
    rooms = [None] * n
    assigned = []
    for op, _payload in rs:
        if op == 'I':
            length = _payload
            pos = next(idx for idx in range(n) if all(x is None for x in rooms[idx:(idx + length)]))
            rooms[pos:(pos + length)] = [len(assigned)] * length
            assigned.append((pos, length))
        else:
            index = _payload
            pos, length = assigned[index]
            assigned[index] = (-1, -1)
            rooms[pos:(pos + length)] = [None] * length
    ans = sum(idx * l for idx, (l, r) in enumerate(assigned) if l != -1)
    return ans


import random
import pickle
import flutes


def main():
    random.seed(flutes.__MAGIC__)
    n = 10
    rs = [('I', 3), ('I', 3), ('O', 0), ('I', 4), ('I', 2), ('O', 2), ('I', 2)]
    with open("data/sl_grand_hotel.pkl", "rb") as f:
        n, rs = pickle.load(f)

    n = int(1e8)
    rs = []
    reservations = set()
    r_cnt = 0
    for idx in range(100000):
        choices = []
        if len(reservations) > 0: choices.append('O')
        if len(reservations) < 100: choices.append('I')
        if random.choice(choices) == 'I':
            l = random.randint(1, n // 100)
            rs.append(('I', l))
            reservations.add(r_cnt)
            r_cnt += 1
        else:
            idx = random.choice(list(reservations))
            rs.append(('O', idx))
            reservations.remove(idx)

    with flutes.work_in_progress():
        print(solution(n, rs))
    # with flutes.work_in_progress():
    #     print(solution_brute(n, rs))


if __name__ == '__main__':
    main()
