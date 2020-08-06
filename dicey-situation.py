import itertools
import math
from typing import Dict, Iterable, List, Tuple


def product(xs: Iterable[float]) -> float:
    ret = 1.
    for x in xs:
        ret *= x
    return ret


def cmp(a, b):
    if a > b: return 1
    if a < b: return -1
    return 0


def solution(dice_vals: List[int], n_dice: int, n_rolls: int):
    dice_vals = sorted(dice_vals)
    n_face = len(dice_vals)

    # Alice: Targets maximum sum. Each dice is independent. Reroll whenever outcome is less than the expected value.
    p_alice_dice: List[List[float]] = [[]]  # p_alice_dice[r][f]: prob. of ending up with face f after r rolls
    p_alice_dice.append([1 / n_face] * n_face)
    for r in range(2, n_rolls + 1):
        probs = [0.] * n_face
        p_reroll = 0
        expected = sum(prob * val for prob, val in zip(p_alice_dice[-1], dice_vals))
        for face in range(n_face):
            if dice_vals[face] <= expected:
                p_reroll += p_alice_dice[-1][face]
            else:
                probs[face] += p_alice_dice[-1][face]
        for face in range(n_face):
            probs[face] += p_reroll / n_face
        p_alice_dice.append(probs)

    max_sum = dice_vals[-1] * n_dice
    p_alice_sum = [1.] + [0.] * max_sum
    for _ in range(n_dice):
        next_p_alice_sum = [0.] * (max_sum + 1)
        for s in range(max_sum + 1):
            for prob, val in zip(p_alice_dice[n_rolls], dice_vals):
                if s - val >= 0:
                    next_p_alice_sum[s] += p_alice_sum[s - val] * prob
        p_alice_sum = next_p_alice_sum
    assert math.isclose(sum(p_alice_sum), 1.)

    # Bob: Always reroll dices with lowest face value. Targets maximum expected payout.
    # State is represented as a tuple of #{face values}, indicating #dices with each face
    # > p_roll_state[d][state]: prob. of getting `state` after rolling d dice
    p_roll_state: List[Dict[Tuple[int, ...], float]] = []
    for n_dice_to_roll in range(n_dice + 1):
        p_cur_state = {}
        p_base_coef = (1 / n_face) ** n_dice_to_roll * math.factorial(n_dice_to_roll)
        for seps in itertools.combinations(range(n_dice_to_roll + n_face - 1), n_face - 1):
            seps = [-1] + list(seps) + [n_dice_to_roll + n_face - 1]
            state = tuple(r - l - 1 for l, r in zip(seps, seps[1:]))
            prob = p_base_coef / product(math.factorial(x) for x in state)
            p_cur_state[state] = prob
        assert math.isclose(sum(p_cur_state.values()), 1.)
        p_roll_state.append(p_cur_state)
    state_sum: Dict[Tuple[int, ...], int] = {}
    for state in p_roll_state[n_dice].keys():
        sum_val = sum(cnt * val for cnt, val in zip(state, dice_vals))
        state_sum[state] = sum_val
    ans = 0.
    for alice_val, alice_prob in enumerate(p_alice_sum):
        if alice_prob == 0.: continue
        # > exp_bob[(#face0, #face1, ...)]: expected outcome for Bob given state
        exp_bob = prev_exp_bob = {state: cmp(sum_val, alice_val) for state, sum_val in state_sum.items()}
        for r in range(2, n_rolls + 1):
            exp_bob = {}
            for state in p_roll_state[n_dice].keys():
                choices = []
                next_state = list(state)
                idx = 0
                for n_reroll in range(n_dice + 1):
                    if n_reroll > 0:
                        while next_state[idx] == 0: idx += 1
                        next_state[idx] -= 1
                    expected = 0.
                    for roll_state, roll_prob in p_roll_state[n_reroll].items():
                        potential_state = tuple(a + b for a, b in zip(next_state, roll_state))
                        expected += prev_exp_bob[potential_state] * roll_prob
                    choices.append(expected)
                print(alice_val, alice_prob, r, state, choices)
                exp_bob[state] = max(choices)
            prev_exp_bob = exp_bob
        cur_exp = sum(prob * exp_bob[state] for state, prob in p_roll_state[n_dice].items())
        ans += alice_prob * cur_exp

    """
    for alice_val, alice_prob in enumerate(p_alice_sum):
        if alice_prob == 0.: continue
        # > exp_bob[(#face0, #face1, ...)]: prob. of state in current roll
        exp_bob = p_roll_state[n_dice]
        for r in range(2, n_rolls + 1):
            next_p_bob = defaultdict(float)
            for state, prob in exp_bob:
                sum_val = state_sum[state]
                idx = 0
                state = list(state)
                for n_reroll in range(n_dice):
                    if sum_val > alice_val: break  # reroll just enough dice to be greater than Alice's sum
                    while state[idx] == 0: idx += 1
                    new_sum_val = sum_val - dice_vals[idx] + expected_face[n_rolls - r + 1]
                    # > if rerolling this dice leads to worse results, then prefer a tie instead
                    if new_sum_val < sum_val == alice_val: break
                    state[idx] -= 1
                else:
                    n_reroll = n_dice
                for roll_state, roll_prob in p_roll_state[n_reroll]:
                    next_state = tuple(a + b for a, b in zip(state, roll_state))
                    next_p_bob[next_state] += prob * roll_prob
            exp_bob = list(next_p_bob.items())
        cur = 0.
        print(alice_val, alice_prob, exp_bob)
        for state, bob_prob in exp_bob:
            bob_val = state_sum[state]
            cur += (1 if bob_val > alice_val else -int(bob_val < alice_val)) * bob_prob
        ans += alice_prob * cur
    """
    return f"{ans:.7f}"[2:]


import pickle
import random
import flutes


def main():
    random.seed(flutes.__MAGIC__)
    dice_vals = [1, 2, 3]
    d = 1
    r = 2

    with open("data/sl_dicey_situation.pkl", "rb") as f:
        dice_vals, d, r = pickle.load(f)

    # dice_vals = [random.randint(1, 20) for _ in range(6)]
    # d = 8
    # r = 8

    dice_vals = [1, 2, 3]
    d = 8
    r = 8

    print(dice_vals, d, r)

    with flutes.work_in_progress():
        print(solution(dice_vals, d, r))


if __name__ == '__main__':
    main()
