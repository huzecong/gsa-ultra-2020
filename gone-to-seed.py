import itertools
import math
from typing import Dict, Iterator, Tuple

from tqdm import tqdm

profile = lambda x: x


class Frac:
    def __init__(self, numerator: int, denominator: int = 1):
        self.numerator = numerator
        self.denominator = denominator

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

    def simplify(self):
        g = math.gcd(self.denominator, self.numerator)
        if g > 1:
            self.numerator //= g
            self.denominator //= g
        return self


@profile
def solution(_skills: Dict[str, int]) -> str:
    skills = [_skills["Andy"]] + [v for k, v in _skills.items() if k != "Andy"]
    win_prob = [[Frac(a, a + b).simplify() for b in skills] for a in skills]
    n = len(skills)

    dp_vals = [0] * (n << n)
    for a in range(n):
        for b in range(n):
            if a == b: continue
            dp_vals[a << n | 1 << a | 1 << b] = win_prob[a][b]
    for n_players in itertools.takewhile(lambda i: i <= n, (1 << i for i in itertools.count(2))):
        cnt = math.factorial(n_players - 1) // math.factorial(n_players >> 1) // math.factorial((n_players >> 1) - 1)
        for state_players in itertools.combinations(range(n), n_players):
            state = sum(1 << x for x in state_players)
            candidates = [0] if state & 1 else state_players
            for winner in candidates:
                players = [x for x in state_players if x != winner]
                prob = 0
                for other_half in itertools.combinations(players, n_players >> 1):
                    other_state = sum(1 << x for x in other_half)
                    cur_state = state - other_state
                    cur_prob = 0
                    for other_winner in other_half:
                        cur_prob += dp_vals[other_winner << n | other_state] * win_prob[winner][other_winner]
                    cur_prob *= dp_vals[winner << n | cur_state]
                    prob += cur_prob
                prob /= cnt
                dp_vals[winner << n | state] = prob.simplify()

    ans = dp_vals[(1 << n) - 1]
    return str(ans.numerator) + str(ans.denominator)


from collections import defaultdict
from fractions import Fraction


def product(xs):
    ret = 1
    for x in xs:
        ret *= x
    return ret


State = Tuple[Tuple[int, int], ...]


def gen_perms(n: int) -> Iterator[Tuple[State, Fraction]]:
    perm_plans = math.factorial(n) // (2 ** (n // 2)) // (n // 2)  # math.factorial(n // 2 - 1)
    # print(perm_plans)
    prob = Fraction(1, perm_plans)
    boxes = [[None, None] for _ in range(n // 2)]
    cnt = 0

    def dfs(x):
        if x == n:
            nonlocal cnt
            pairs = list(map(tuple, boxes))
            for perm in itertools.permutations(pairs[1:]):
                cnt += 1
                yield (pairs[0],) + perm, prob
            return
        for box in boxes:
            if box[0] is None:
                box[0] = x
                yield from dfs(x + 1)
                box[0] = None
                break
            elif box[1] is None:
                box[1] = x
                yield from dfs(x + 1)
                box[1] = None

    yield from dfs(0)
    assert cnt == perm_plans


def solution_search(_skills: Dict[str, int]) -> str:
    a = [_skills["Andy"]] + [v for k, v in _skills.items() if k != "Andy"]
    n = len(a)

    def get_iter(cur_size: int) -> Iterator[Tuple[State, Fraction]]:
        if cur_size == n: return gen_perms(n)
        return probs.items()

    size = n
    probs: Dict[State, Fraction] = {}
    while size > 2:
        next_probs: Dict[State, Fraction] = defaultdict(Fraction)
        for state, prob in get_iter(size):
            assert state[0][0] == 0
            denominator = product(a[pk[0]] + a[pk[1]] for pk in state)
            for choices in itertools.product(*([[0, 1]] * (size // 2 - 1))):
                next_players = [0] + [pk[choice] for pk, choice in zip(state[1:], choices)]
                numerator = product(a[x] for x in next_players)
                new_state = []
                for i in range(0, len(next_players), 2):
                    x, y = next_players[i], next_players[i + 1]
                    if x > y: x, y = y, x
                    new_state.append((x, y))
                # new_state.sort()
                next_probs[tuple(new_state)] += Fraction(numerator, denominator) * prob

        probs = next_probs
        size //= 2
    ans = Fraction(0)
    for state, prob in get_iter(2):
        assert state[0][0] == 0
        ans += prob * Fraction(a[0], a[0] + a[state[0][1]])

    return str(ans.numerator) + str(ans.denominator)


def solution_brute(_skills: Dict[str, int]) -> str:
    a = [_skills["Andy"]] + [v for k, v in _skills.items() if k != "Andy"]
    n = len(a)

    ans = Fraction(0)
    for state in tqdm(itertools.permutations(range(n)), total=math.factorial(n)):
        for choices in itertools.product(*([[0, 1]] * (n - 1))):
            cur_prob = Fraction(1)
            pos = 0
            cur_state = state
            while len(cur_state) > 1:
                cur_choices = choices[pos:(pos + len(cur_state) // 2)]
                pos += len(cur_state) // 2

                next_state = tuple(cur_state[idx * 2 + choice] for idx, choice in enumerate(cur_choices))
                if 0 not in next_state:
                    cur_prob = 0
                    break
                numerator = product(a[x] for x in next_state)
                denominator = product(a[cur_state[i]] + a[cur_state[i + 1]] for i in range(0, len(cur_state), 2))
                cur_prob *= Fraction(numerator, denominator)
                cur_state = next_state
            ans += cur_prob
    ans /= math.factorial(n)
    # ans /= 2 ** (n - 1)
    return str(ans.numerator) + str(ans.denominator)


import random
import pickle
import flutes


def main():
    random.seed(flutes.__MAGIC__)
    skills = {'Andy': 7, 'Novak': 5, 'Roger': 3, 'Rafael': 2}

    with open("data/sl_gone_to_seed.pkl", "rb") as f:
        skills, = pickle.load(f)

    n = 8
    values = [random.randint(1, 20) for _ in range(n)]
    names = ['Andy'] + [chr(65 + x) for x in range(1, n)]
    skills = dict(zip(names, values))

    print(skills)

    with flutes.work_in_progress():
        print(solution(skills))
    with flutes.work_in_progress():
        print(solution_search(skills))
    # with flutes.work_in_progress():
    #     print(solution_brute(skills))


if __name__ == '__main__':
    main()
