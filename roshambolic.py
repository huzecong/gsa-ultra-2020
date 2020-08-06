from collections import deque


def solution(a: str, b: str):
    qa = list(reversed(a))
    qb = list(reversed(b))
    for a, b in zip(qa, qb):
        if a == b:
            qa.append(a)
            qb.append(b)
        elif a + b in {"RS", "SP", "PR"}:
            qa += [b, a]
        else:
            qb += [a, b]
    return min(len(qa), len(qb))


def solution_ref(a: str, b: str):
    qa = deque(a)
    qb = deque(b)
    ans = 0
    while len(qa) > 0 and len(qb) > 0:
        ans += 1
        s = qa.pop() + qb.pop()
        if s[0] == s[1]:
            qa.appendleft(s[0])
            qb.appendleft(s[1])
        elif s in ("RS", "SP", "PR"):
            qa.appendleft(s[1])
            qa.appendleft(s[0])
        else:
            qb.appendleft(s[0])
            qb.appendleft(s[1])
    return ans


def solution_rev(a: str, b: str):
    qa = deque(reversed(a))
    qb = deque(reversed(b))
    ans = 0
    while len(qa) > 0 and len(qb) > 0:
        ans += 1
        a, b = qa.popleft(), qb.popleft()
        if a == b:
            qa.append(a)
            qb.append(b)
        elif a + b in ("RS", "SP", "PR"):
            qa.append(b)
            qa.append(a)
        else:
            qb.append(a)
            qb.append(b)
    return ans


import random
import flutes


def main():
    random.seed(flutes.__MAGIC__)
    a = "SRR"
    b = "PPS"
    n = 1000
    a = "".join(random.choice("RSP") for _ in range(n))
    b = "".join(random.choice("RSP") for _ in range(n))
    with flutes.work_in_progress():
        print(solution(a, b))
    with flutes.work_in_progress():
        print(solution_ref(a, b))
    with flutes.work_in_progress():
        print(solution_rev(a, b))


if __name__ == '__main__':
    main()
