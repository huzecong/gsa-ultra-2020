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


# @profile
def solution(dice_vals: List[int], n_dice: int, n_rolls: int):
    dice_vals = sorted(dice_vals)
    n_face = len(dice_vals)

    # State is represented as a tuple of #{face values}, indicating #dices with each face
    # > p_roll_state[d][state]: prob. of getting `state` after rolling d dice
    p_roll_state: List[List[Tuple[int, float]]] = []
    max_state = n_dice << (3 * (n_face - 1))
    state_sum: Dict[int, int] = {}
    state_transitions: List[List[int]] = [0] * (max_state + 1)
    for n_dice_to_roll in range(n_dice + 1):
        p_cur_state = []
        p_base_coef = (1 / n_face) ** n_dice_to_roll * math.factorial(n_dice_to_roll)
        for seps in itertools.combinations(range(n_dice_to_roll + n_face - 1), n_face - 1):
            seps = [-1] + list(seps) + [n_dice_to_roll + n_face - 1]
            state = [r - l - 1 for l, r in zip(seps, seps[1:])]
            state_id = sum(x << (3 * i) for i, x in enumerate(state))
            prob = p_base_coef / product(math.factorial(x) for x in state)
            p_cur_state.append((state_id, prob))

            if n_dice_to_roll == n_dice:
                sum_val = sum(cnt * val for cnt, val in zip(state, dice_vals))
                state_sum[state_id] = sum_val
                transitions = []
                next_state_id = state_id
                idx = 0
                for n_reroll in range(1, n_dice):
                    while state[idx] == 0: idx += 1
                    state[idx] -= 1
                    next_state_id -= 1 << (3 * idx)
                    transitions.append(next_state_id)
                state_transitions[state_id] = transitions

        assert math.isclose(sum(p for _, p in p_cur_state), 1.)
        if n_dice_to_roll == n_dice:
            p_cur_state.sort(key=lambda xs: state_sum[xs[0]])
        p_roll_state.append(p_cur_state)

    max_sum = dice_vals[-1] * n_dice
    valid_sums = [True] + [False] * max_sum
    for _ in range(n_dice):
        next_valid_sum = [False] * (max_sum + 1)
        for s in range(max_sum + 1):
            if not valid_sums[s]: continue
            for val in dice_vals:
                if s + val <= max_sum:
                    next_valid_sum[s + val] = True
        valid_sums = next_valid_sum

    # Bob: Always reroll dices with lowest face value. Targets maximum expected payout.
    expected_payout: Dict[int, float] = {}  # expected payout for Bob given Alice's sum
    for alice_val in range(max_sum + 1):
        if not valid_sums[alice_val]: continue
        # > exp_bob[(#face0, #face1, ...)]: expected outcome for Bob given state
        exp_bob = prev_exp_bob = {state: cmp(sum_val, alice_val) for state, sum_val in state_sum.items()}
        for r in range(2, n_rolls + 1):
            exp_bob = prev_exp_bob.copy()
            trans_cache = {}
            exp_reroll_all = 0.
            for state, prob in p_roll_state[n_dice]:
                exp_reroll_all += prev_exp_bob[state] * prob
            for state, _ in p_roll_state[n_dice]:
                if state_sum[state] > alice_val: break
                choices = [prev_exp_bob[state]]
                for n_reroll, next_state in enumerate(state_transitions[state]):
                    if next_state in trans_cache:
                        choices.append(trans_cache[next_state])
                    else:
                        expected = 0.
                        for roll_state, roll_prob in p_roll_state[n_reroll + 1]:
                            expected += prev_exp_bob[next_state + roll_state] * roll_prob
                        trans_cache[next_state] = expected
                        choices.append(expected)
                choices.append(exp_reroll_all)
                exp_bob[state] = max(choices)
                # exp_bob[state], idx = max((x, -i) for i, x in enumerate(choices))
                # if idx != 0:
                #     print(r, alice_val, state_sum[state], bin(state)[2:].rjust(3 * n_face, '0'), -idx, choices)
            prev_exp_bob = exp_bob
        cur_exp = sum(prob * exp_bob[state] for state, prob in p_roll_state[n_dice])
        expected_payout[alice_val] = cur_exp

    # Alice
    exp_alice = prev_exp_alice = {state: expected_payout[sum_val] for state, sum_val in state_sum.items()}
    for r in range(2, n_rolls + 1):
        exp_alice = {}
        exp_reroll_all = 0.
        for state, prob in p_roll_state[n_dice]:
            exp_reroll_all += prev_exp_alice[state] * prob
        for state, _ in p_roll_state[n_dice]:
            choices = [prev_exp_alice[state]]
            for n_reroll, next_state in enumerate(state_transitions[state]):
                expected = 0.
                for roll_state, roll_prob in p_roll_state[n_reroll + 1]:
                    potential_state = next_state + roll_state
                    expected += prev_exp_alice[potential_state] * roll_prob
                choices.append(expected)
            choices.append(exp_reroll_all)
            exp_alice[state] = min(choices)
        prev_exp_alice = exp_alice

    ans = sum(prob * exp_alice[state] for state, prob in p_roll_state[n_dice])
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

    dice_vals = [random.randint(1, 20) for _ in range(4)]
    d = 8
    r = 8

    # dice_vals = [2, 3, 4, 5]
    # d = 8
    # r = 8

    print(dice_vals, d, r)

    with flutes.work_in_progress():
        print(solution(dice_vals, d, r))


if __name__ == '__main__':
    main()
