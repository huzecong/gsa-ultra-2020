import cmath
import itertools
import math
from collections import defaultdict
from fractions import Fraction
from typing import List, Optional, Tuple, Iterable, Dict, Any, TypeVar

# profile = lambda x: x

T = TypeVar('T')

import numpy as np


# def poly1d(xs: List[int]) -> np.poly1d:
#     return np.poly1d(np.array([Fraction(x) for x in xs], dtype=np.object))

#
# class bitrev:
#     @staticmethod
#     def _compute(x: int) -> int:
#         x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1)
#         x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2)
#         x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4)
#         x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8)
#         x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16)
#         return x
#
#     _values = defaultdict(_compute)
#
#     def __new__(cls, x: int, bits: int) -> int:
#         return cls._values[x] >> (32 - bits)


def fft(a: List[complex], length: int, bit: int) -> List[complex]:
    result = [0] * length
    for i in range(len(a)):
        result[bitrev(i, bit)] = a[i]
    j = 2
    while j <= length:
        wm = cmath.exp(complex(0, ))
        j <<= 1


def convolve(a: List[int], b: List[int]) -> List[int]:
    length, bit = 1, 0
    while length < len(a) + len(b) - 1:
        length <<= 1
        bit += 1
    a = fft(a, length, bit)
    b = fft(b, length, bit)
    for i in range(length):
        a[i] *= b[i]
    result = ifft(a, length, bit)
    return [int(x) for x in result]


class poly1d:
    def __init__(self, coefs: List[int]):
        end = next((idx for idx in range(len(coefs), 0, -1) if coefs[idx - 1] != 0), 0)
        if end == 0:
            coefs = [0]
        elif end < len(coefs):
            del coefs[end:]
        self.coefs = coefs

    def __add(self, other):
        if isinstance(other, int):
            return poly1d([self.coefs[0] + other] + self.coefs[1:])
        return poly1d([x + y for x, y in itertools.zip_longest(self.coefs, other.coefs, fillvalue=0)])

    __add__ = __radd__ = __add

    def __sub__(self, other):
        if isinstance(other, int):
            return poly1d([self.coefs[0] - other] + self.coefs[1:])
        return poly1d([x - y for x, y in itertools.zip_longest(self.coefs, other.coefs, fillvalue=0)])

    def __rsub__(self, other):
        if isinstance(other, int):
            return poly1d([other - self.coefs[0]] + [-x for x in self.coefs[1:]])
        return poly1d([y - x for x, y in itertools.zip_longest(self.coefs, other.coefs, fillvalue=0)])

    def __mul(self, other):
        if isinstance(other, int):
            return poly1d([x * other for x in self.coefs])
        if len(self.coefs) < 16 or len(other.coefs) < 16:
            coefs = [0] * (len(self.coefs) + len(other.coefs) - 1)
            for i, x in enumerate(self.coefs):
                for j, y in enumerate(other.coefs):
                    coefs[i + j] += x * y
        else:
            coefs = convolve(self.coefs, other.coefs)
        return poly1d(coefs)

    __mul__ = __rmul__ = __mul

    def value(self, x: T) -> T:
        ret = 0
        power = 1
        for coef in self.coefs:
            ret += power * coef
            power *= x
        return ret

    def __pow__(self, power, modulo=None):
        assert power == 2
        return self * self


def solution(wabbits: List[Tuple[int, str]], p_numerator: int, p_denominator: int) -> str:
    types = [typ for _, typ in wabbits]
    edges: List[List[int]] = [[] for _ in range(len(wabbits))]
    for idx, (parent, typ) in enumerate(wabbits):
        if parent == -1: continue
        edges[parent].append(idx)
        edges[idx].append(parent)

    p = poly1d([0, 1])
    q = 1 - p
    one = poly1d([1])
    zero = poly1d([0])
    trans_probs = [
        [q ** 2, 2 * p * q, p ** 2],
        [p * q, q ** 2 + p ** 2, p * q],
        [p ** 2, 2 * p * q, q ** 2],
    ]
    bottom_up_probs = [None] * len(wabbits)
    bottom_up_probs_excluded = [None] * len(wabbits)

    def bottom_up_dfs(x: int, parent: Optional[int] = None) -> List[np.poly1d]:
        # Compute P(subtree | node gene)
        node_prob = [one, one, one]
        prefix_prob = []
        child_trans_probs = []
        for child in edges[x]:
            if child == parent: continue
            prefix_prob.append(node_prob)
            child_prob = bottom_up_dfs(child, x)
            child_trans_prob = [inner_prod_3(trans_probs[i], child_prob) for i in range(3)]
            child_trans_probs.append(child_trans_prob)
            node_prob = piecewise_prod_3(node_prob, child_trans_prob)
        pre_idx = len(prefix_prob) - 1
        suffix_prob = [one, one, one]
        for child in reversed(edges[x]):
            if child == parent: continue
            bottom_up_probs_excluded[child] = piecewise_prod_3(prefix_prob[pre_idx], suffix_prob)
            suffix_prob = piecewise_prod_3(suffix_prob, child_trans_probs[pre_idx])
            pre_idx -= 1
        cur_typ = wabbits[x][1]
        if cur_typ == "R":
            node_prob[2] = zero
            for child in edges[x]:
                if child == parent: continue
                bottom_up_probs_excluded[child][2] = zero
        elif cur_typ == "G":
            node_prob[0] = node_prob[1] = zero
            for child in edges[x]:
                if child == parent: continue
                prob = bottom_up_probs_excluded[child]
                prob[0] = prob[1] = zero
        bottom_up_probs[x] = node_prob
        return node_prob

    post_prob = bottom_up_dfs(0)
    sum_exp = zero

    @profile
    def top_down_dfs(x: int, top_down_prob: List[np.prod], parent: Optional[int] = None) -> None:
        node_trans_prob = [inner_prod_3(trans_probs[i], top_down_prob) for i in range(3)]
        node_prob = piecewise_prod_3(bottom_up_probs[x], node_trans_prob)
        for child in edges[x]:
            if child == parent: continue
            child_prob = piecewise_prod_3(bottom_up_probs_excluded[child], node_trans_prob)
            top_down_dfs(child, child_prob, x)

        if types[x] == "?":
            nonlocal sum_exp
            sum_exp += node_prob[2]

    top_down_dfs(0, [one, one, one])

    p = Fraction(p_numerator, p_denominator)
    ans = sum_exp.value(p)

    natural_prob = [Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)]
    total_prob = sum(prob.value(p) * init for prob, init in zip(post_prob, natural_prob))

    ans /= total_prob * 4
    ans += sum(typ == 'G' for typ in types)
    # print(float(ans))
    # print(float(total_prob))
    return str(ans.numerator) + str(ans.denominator)


def gcd(a, b):
    g = 1
    while a > 0:
        if a & 1:
            if b & 1:
                if a < b: a, b = b, a
                a -= b
            else:
                b >>= 1
                while not (b & 1):
                    b >>= 1
        else:
            if b & 1:
                a >>= 1
                while not (a & 1):
                    a >>= 1
            else:
                a >>= 1
                b >>= 1
                g <<= 1
    assert a == 0
    return b * g


import math
from typing import List, Optional, Tuple


class Frac:
    def __init__(self, numerator: int, denominator: int = 1):
        self.numerator = numerator
        self.denominator = denominator

    @profile
    def __add(self, other: 'Frac') -> 'Frac':
        if isinstance(other, int):
            if other == 0: return self
            return Frac(self.numerator + self.denominator * other, self.denominator)
        a, b = self.denominator, other.denominator
        if a == b:
            return Frac(self.numerator + other.numerator, a)
        d, m = divmod(a, b)
        if m == 0:
            return Frac(self.numerator + other.numerator * d, a)
        return Frac(self.numerator * b + other.numerator * a, a * b)

    __add__ = __radd__ = __add

    @profile
    def __iadd__(self, other: 'Frac') -> 'Frac':
        a, b = self.denominator, other.denominator
        d, m = divmod(a, b)
        if m == 0:
            self.numerator += other.numerator * d
        else:
            self.numerator = self.numerator * b + other.numerator * a
            self.denominator *= b
        return self

    def __mul(self, other: 'Frac') -> 'Frac':
        if isinstance(other, int):
            if other == 1: return self
            return Frac(self.numerator * other, self.denominator)
        return Frac(self.numerator * other.numerator, self.denominator * other.denominator)

    __mul__ = __rmul__ = __mul

    def __imul__(self, other: 'Frac') -> 'Frac':
        self.numerator *= other.numerator
        self.denominator *= other.denominator
        return self

    def __truediv__(self, other: 'Frac') -> 'Frac':
        if isinstance(other, int):
            return Frac(self.numerator, self.denominator * other)
        return Frac(self.numerator * other.denominator, self.denominator * other.numerator)

    def __repr__(self):
        return f"{self.numerator}/{self.denominator}"

    def __pow__(self, power, modulo=None):
        assert power == 2
        return self * self

    def simplify(self):
        # g = gcd(self.denominator, self.numerator)
        assert g == math.gcd(self.denominator, self.numerator)
        if g > 1:
            self.numerator //= g
            self.denominator //= g
        return self


def inner_prod_3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def piecewise_prod_3(a, b):
    return [a[0] * b[0], a[1] * b[1], a[2] * b[2]]


@profile
def solution_frac(wabbits: List[Tuple[int, str]], p_numerator: int, p_denominator: int) -> str:
    p = Frac(p_numerator, p_denominator)
    q = Frac(p_denominator - p_numerator, p_denominator)
    types = [typ for _, typ in wabbits]
    edges: List[List[int]] = [[] for _ in range(len(wabbits))]
    for idx, (parent, typ) in enumerate(wabbits):
        if parent == -1: continue
        edges[parent].append(idx)
        edges[idx].append(parent)

    zero = Frac(0)
    one = Frac(1)
    trans_probs = [
        [q ** 2, 2 * p * q, p ** 2],
        [p * q, q ** 2 + p ** 2, p * q],
        [p ** 2, 2 * p * q, q ** 2],
    ]
    for xs in trans_probs:
        for x in xs:
            x.simplify()
    bottom_up_probs = [None] * len(wabbits)
    child_trans_probs = [None] * len(wabbits)

    @profile
    def bottom_up_dfs(x: int, parent: Optional[int] = None) -> List[Frac]:
        # Compute P(subtree | node gene)
        node_prob = [one, one, one]
        for child in edges[x]:
            if child == parent: continue
            child_prob = bottom_up_dfs(child, x)
            child_trans_probs[child] = child_trans_prob = [inner_prod_3(trans_probs[i], child_prob) for i in range(3)]
            node_prob = piecewise_prod_3(node_prob, child_trans_prob)
        cur_typ = wabbits[x][1]
        if cur_typ == "R":
            # node_prob[0].simplify()
            # node_prob[1].simplify()
            node_prob[2] = zero
        elif cur_typ == "G":
            # node_prob[2].simplify()
            node_prob[0] = node_prob[1] = zero
        else:
            # node_prob[0].simplify()
            # node_prob[1].simplify()
            # node_prob[2].simplify()
            pass
        bottom_up_probs[x] = node_prob
        return node_prob

    bottom_up_dfs(0)
    natural_prob = [Frac(1, 4), Frac(1, 2), Frac(1, 4)]
    total_prob = inner_prod_3(bottom_up_probs[0], natural_prob)
    ans = 0

    @profile
    def top_down_dfs(x: int, depth: int, top_down_prob: List[Frac], parent: Optional[int] = None) -> None:
        node_prob = [(bottom_up_probs[x][i] * inner_prod_3(trans_probs[i], top_down_prob))
                     for i in range(3)]
        if depth % 8 == 0:
            for i in range(3):
                node_prob[i].simplify()
        for child in edges[x]:
            if child == parent: continue
            child_trans_prob = child_trans_probs[child]
            child_prob = [(node_prob[i] / child_trans_prob[i]) for i in range(3)]
            # child_prob[0].simplify()
            # child_prob[1].simplify()
            # child_prob[2].simplify()
            top_down_dfs(child, depth + 1, child_prob, x)

        if types[x] == "?":
            nonlocal ans
            ans += node_prob[2]

    top_down_dfs(0, 0, [(one / (4 * total_prob))] * 3)
    # ans /= total_prob * 4
    ans += sum(typ == 'G' for typ in types)
    ans.simplify()
    return str(ans.numerator) + str(ans.denominator)


def solution_frac_builtin(wabbits: List[Tuple[int, str]], p_numerator: int, p_denominator: int) -> str:
    from fractions import Fraction as Frac
    p = Frac(p_numerator, p_denominator)
    q = Frac(p_denominator - p_numerator, p_denominator)
    types = [typ for _, typ in wabbits]
    edges: List[List[int]] = [[] for _ in range(len(wabbits))]
    for idx, (parent, typ) in enumerate(wabbits):
        if parent == -1: continue
        edges[parent].append(idx)
        edges[idx].append(parent)

    trans_probs = [
        [q ** 2, 2 * p * q, p ** 2],
        [p * q, q ** 2 + p ** 2, p * q],
        [p ** 2, 2 * p * q, q ** 2],
    ]
    bottom_up_probs = [None] * len(wabbits)
    bottom_up_probs_excluded = [None] * len(wabbits)

    def bottom_up_dfs(x: int, parent: Optional[int] = None) -> List[Frac]:
        # Compute P(subtree | node gene)
        node_prob = [1, 1, 1]
        # prefix_prob = []
        # child_trans_probs = []
        for child in edges[x]:
            if child == parent: continue
            # prefix_prob.append(node_prob)
            child_prob = bottom_up_dfs(child, x)
            child_trans_prob = [inner_prod_3(trans_probs[i], child_prob) for i in range(3)]
            # child_trans_probs.append(child_trans_prob)
            node_prob = piecewise_prod_3(node_prob, child_trans_prob)
        # pre_idx = len(prefix_prob) - 1
        # suffix_prob = [1, 1, 1]
        # for child in reversed(edges[x]):
        #     if child == parent: continue
        #     bottom_up_probs_excluded[child] = piecewise_prod_3(prefix_prob[pre_idx], suffix_prob)
        #     # assert piecewise_prod_3(bottom_up_probs_excluded[child], child_trans_probs[pre_idx]) == node_prob
        #     suffix_prob = piecewise_prod_3(suffix_prob, child_trans_probs[pre_idx])
        #     pre_idx -= 1
        cur_typ = wabbits[x][1]
        if cur_typ == "R":
            node_prob[2] = 0
            # for child in edges[x]:
            #     if child == parent: continue
            #     bottom_up_probs_excluded[child][2] = 0
        elif cur_typ == "G":
            node_prob[0] = node_prob[1] = 0
            # for child in edges[x]:
            #     if child == parent: continue
            #     prob = bottom_up_probs_excluded[child]
            #     prob[0] = prob[1] = 0
        bottom_up_probs[x] = node_prob
        return node_prob

    natural_prob = [Frac(1, 4), Frac(1, 2), Frac(1, 4)]
    total_prob = inner_prod_3(bottom_up_dfs(0), natural_prob)
    ans = 0

    def top_down_dfs(x: int, top_down_prob: List[Frac], parent: Optional[int] = None) -> None:
        node_prob = [bottom_up_probs[x][i] * inner_prod_3(trans_probs[i], top_down_prob)
                     for i in range(3)]
        for child in edges[x]:
            if child == parent: continue
            # child_prob = [bottom_up_probs_excluded[child][i] * inner_prod_3(trans_probs[i], top_down_prob)
            #               for i in range(3)]
            child_prob = [node_prob[i] / sum(trans_probs[i][j] * bottom_up_probs[child][j] for j in range(3))
                          for i in range(3)]
            # assert child_prob == child_prob2
            top_down_dfs(child, child_prob, x)

        if types[x] == "?":
            nonlocal ans
            ans += node_prob[2]

    top_down_dfs(0, (1, 1, 1))
    ans /= total_prob * 4
    ans += sum(typ == 'G' for typ in types)
    # print(float(ans))
    # print(float(total_prob))
    return str(ans.numerator) + str(ans.denominator)


def solution_n2(wabbits: List[Tuple[int, str]], p_numerator: int, p_denominator: int) -> str:
    from fractions import Fraction as Frac
    p = Frac(p_numerator, p_denominator)
    types = [typ for _, typ in wabbits]
    children: List[List[int]] = [[] for _ in range(len(wabbits))]
    edges: List[List[int]] = [[] for _ in range(len(wabbits))]
    for idx, (parent, typ) in enumerate(wabbits):
        if parent == -1: continue
        edges[parent].append(idx)
        edges[idx].append(parent)
        children[parent].append(idx)

    trans_probs = [
        [(1 - p) ** 2, 2 * p * (1 - p), p ** 2],
        [p * (1 - p), (1 - p) ** 2 + p ** 2, p * (1 - p)],
        [p ** 2, 2 * p * (1 - p), (1 - p) ** 2],
    ]

    def dfs(x: int, parent: Optional[int] = None) -> List[Frac]:
        # Compute P(subtree | node gene)
        probs = [Frac(1), Frac(1), Frac(1)]
        for child in edges[x]:
            if child == parent: continue
            prob = dfs(child, x)
            probs = [probs[i] * sum(trans_probs[i][j] * prob[j] for j in range(3)) for i in range(3)]
        cur_typ = wabbits[x][1]
        if cur_typ == "R":
            probs[2] = 0
        elif cur_typ == "G":
            probs[0] = probs[1] = 0
        return probs

    total_prob = sum(prob * init for prob, init in zip(dfs(0), [Frac(1, 4), Frac(1, 2), Frac(1, 4)]))
    # print(total_prob)
    ans = 0
    for idx, (_, typ) in enumerate(wabbits):
        if typ != "?": continue
        cur_prob = dfs(idx)[2] / 4
        # print(idx, cur_prob)
        ans += cur_prob
    ans /= total_prob
    ans += Frac(sum(typ == 'G' for typ in types))
    # print(ans)
    return str(ans.numerator) + str(ans.denominator)


def solution_brute(wabbits: List[Tuple[int, str]], p_numerator: int, p_denominator: int) -> str:
    from fractions import Fraction as Frac
    p = Frac(p_numerator, p_denominator)
    init_prob = [Frac(1, 4), Frac(1, 2), Frac(1, 4)]
    trans_probs = [
        [(1 - p) ** 2, 2 * p * (1 - p), p ** 2],
        [p * (1 - p), (1 - p) ** 2 + p ** 2, p * (1 - p)],
        [p ** 2, 2 * p * (1 - p), (1 - p) ** 2],
    ]
    candidates: List[Tuple[int, ...]] = []
    for _, typ in wabbits:
        if typ == "R":
            candidates.append((0, 1))
        elif typ == "G":
            candidates.append((2,))
        else:
            candidates.append((0, 1, 2))
    sum_exp = Frac(0)
    sum_prob = Frac(0)
    for genes in itertools.product(*candidates):
        prob = init_prob[genes[0]]
        for g, (p, _) in zip(genes[1:], wabbits[1:]):
            prob *= trans_probs[genes[p]][g]
        # print(genes, prob)
        sum_exp += prob * sum(g == 2 for g in genes)
        sum_prob += prob
    # print(sum_prob)
    ans = sum_exp / sum_prob
    # print(ans)
    return str(ans.numerator) + str(ans.denominator)


import random
import pickle
import flutes


def main():
    import sys
    sys.setrecursionlimit(32768)
    random.seed(flutes.__MAGIC__)
    # wabbits = [(-1, '?'), (0, 'R')]
    wabbits = [(-1, '?'), (0, 'R'), (0, 'R')]
    wabbits = [(-1, 'R'), (0, 'R'), (1, '?')]
    wabbits = [(-1, '?'), (0, 'R'), (1, 'R')]
    p_numerator = 1
    p_denominator = 5
    # with open("data/sl_wascally_wabbits.pkl", "rb") as f:
    #     wabbits, p_numerator, p_denominator = pickle.load(f)

    n = 10000
    types = [random.choice("?RG") for _ in range(n)]
    # parent = [-1] + [random.randint(0, idx) for idx in range(n - 1)]
    parent = [-1] + list(range(n - 1))
    # parent = [-1] + list(range(n - 1))
    wabbits = list(zip(parent, types))
    p_numerator = 49
    p_denominator = 100

    # print(wabbits)
    # print(p_numerator, p_denominator)

    # with flutes.work_in_progress():
    #     print(solution(wabbits, p_numerator, p_denominator))
    with flutes.work_in_progress():
        print(solution_frac(wabbits, p_numerator, p_denominator))
    with flutes.work_in_progress():
        print(solution_frac_builtin(wabbits, p_numerator, p_denominator))
    with flutes.work_in_progress():
        print(solution_n2(wabbits, p_numerator, p_denominator))
    with flutes.work_in_progress():
        print(solution_brute(wabbits, p_numerator, p_denominator))


if __name__ == '__main__':
    main()
