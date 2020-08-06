import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Union


def tokenize(s: str) -> List[str]:
    pos = 0
    tokens = []
    while pos < len(s):
        if s[pos].isspace():
            pos += 1
        elif s[pos] in {'(', ')'}:
            tokens.append(s[pos])
            pos += 1
        else:
            assert s[pos].isalpha()
            end = next((idx for idx in range(pos + 1, len(s)) if not s[idx].isalpha()), len(s))
            tokens.append(s[pos:end])
            pos = end
    return tokens


class Node:
    pass


@dataclass(unsafe_hash=True)
class And(Node):
    children: Tuple[Node, ...]

    def __bool__(self):
        return all(self.children)

    def __str__(self):
        return "(" + " and ".join(map(str, self.children)) + ")"


@dataclass(unsafe_hash=True)
class Or(Node):
    children: Tuple[Node, ...]

    def __bool__(self):
        return any(self.children)

    def __str__(self):
        return "(" + " or ".join(map(str, self.children)) + ")"


@dataclass(unsafe_hash=True)
class Not(Node):
    child: Node

    def __bool__(self):
        return not bool(self.child)

    def __str__(self):
        return f"not {self.child}"


@dataclass(unsafe_hash=True)
class Var(Node):
    idx: int
    var_list: List[bool] = field(repr=False, hash=False, default=None)

    def __bool__(self):
        return self.var_list[self.idx]

    def __str__(self):
        return chr(65 + self.idx)


def parse(tokens: List[str]) -> Tuple[Node, List[bool]]:
    stack = []
    var_list = [False] * 26

    def reduce():
        op, b = stack[-2:]
        if op == 'NOT':
            node = Not(b)
            del stack[-1]
        else:
            a = stack[-3]
            op_typ = And if op == 'AND' else Or
            if type(b) is op_typ:
                b.children.append(a)
                node = b
            else:
                node = op_typ([a, b])
            del stack[-2:]
        stack[-1] = node

    def reduce_if(*ops):
        while len(stack) > 1 and stack[-2] in ops:
            reduce()

    for token in tokens:
        if token == ')':
            while stack[-2] != '(':
                reduce()
            stack[-2] = stack[-1]
            del stack[-1]
        elif token in {"AND", "NOT", "OR", "("}:
            if token == "OR":
                reduce_if("AND", "NOT")
            elif token == "AND":
                reduce_if("NOT")
            stack.append(token)
        else:
            stack.append(Var(ord(token) - 65, var_list))

    while len(stack) > 1:
        reduce()

    def convert(x: Node) -> None:
        if isinstance(x, (And, Or)):
            x.children = tuple(x.children)
            for y in x.children:
                convert(y)
        elif isinstance(x, Not):
            convert(x.child)

    convert(stack[0])
    return stack[0], var_list


def simplify(tree: Node) -> Union[Node, bool]:
    if isinstance(tree, Not):
        child = simplify(tree.child)
        if isinstance(child, bool):
            return not child
        if isinstance(child, Not):
            return child.child
        return Not(child)
    if isinstance(tree, (And, Or)):
        children = list(set(simplify(x) for x in tree.children))
        if isinstance(tree, And):
            children = [x for x in children if x is not True]
            if any(x is False for x in children):
                return False
            variables = set(x.idx for x in children if isinstance(x, Var))
            if any(isinstance(x, Not) and isinstance(x.child, Var) and x.child.idx in variables for x in children):
                return False
            filtered_children = []
            for child in children:
                if isinstance(child, And):
                    filtered_children.extend(child.children)
                    continue
                if isinstance(child, Or):
                    if any(isinstance(x, Var) and x.idx in variables for x in child.children):
                        continue
                    nodes = [x for x in child.children if
                             not (isinstance(x, Not) and isinstance(x.child, Var) and x.child.idx in variables)]
                    if len(nodes) == 0:
                        continue
                    elif len(nodes) == 1:
                        child = nodes[0]
                    else:
                        child = Or(tuple(nodes))
                filtered_children.append(child)
            return And(tuple(filtered_children))
        else:
            children = [x for x in children if x is not False]
            if any(x is True for x in children):
                return True
            variables = set(x.idx for x in children if isinstance(x, Var))
            if any(isinstance(x, Not) and isinstance(x.child, Var) and x.child.idx in variables for x in children):
                return True
            filtered_children = []
            for child in children:
                if isinstance(child, Or):
                    filtered_children.extend(child.children)
                    continue
                if isinstance(child, And):
                    if any(isinstance(x, Var) and x.idx in variables for x in child.children):
                        continue
                    nodes = [x for x in child.children if
                             not (isinstance(x, Not) and isinstance(x.child, Var) and x.child.idx in variables)]
                    if len(nodes) == 0:
                        continue
                    elif len(nodes) == 1:
                        child = nodes[0]
                    else:
                        child = And(tuple(nodes))
                filtered_children.append(child)
            return Or(tuple(filtered_children))
    return tree


def solution(n: int, prop: str) -> int:
    tokens = tokenize(prop)
    # print(tokens)
    tree, var_list = parse(tokens)
    print(tree)
    tree = simplify(tree)
    print(tree)
    variables = set()

    def dfs(node: Node):
        if isinstance(node, Var):
            variables.add(node.idx)
        elif isinstance(node, Not):
            dfs(node.child)
        else:
            for x in node.children:
                dfs(x)

    ans = 0
    # print(str(tree))
    ast = compile(str(tree), "", "eval")
    dfs(tree)
    variables = list(variables)
    var_names = [chr(65 + i) for i in variables]
    for vals in itertools.product(*([[False, True]] * len(variables))):
        # for idx, val in zip(variables, vals):
        #     var_list[idx] = val
        # assert eval(ast, dict(zip(var_names, vals))) == bool(tree)
        # if bool(tree): ans += 1
        ans += eval(ast, dict(zip(var_names, vals)))
    ans *= pow(2, n - len(variables))
    return ans


def solution_brute(n: int, prop: str) -> int:
    prop = prop.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
    ast = compile(prop, "", "eval")
    ans = 0
    for state in range(1 << n):
        variables = {chr(65 + i): state >> i & 1 for i in range(n)}
        ans += eval(ast, variables)
    return ans


import random
import pickle
import flutes
import sys


def generate(n, size: int) -> Node:
    choices = []
    if size == 1:
        choices.append(Var)
    if size >= 2:
        choices.append(Not)
    if size >= 3:
        choices.extend([And, Or])
    op = random.choice(choices)
    if op is Var:
        return Var(random.randint(0, n - 1))
    elif op is Not:
        return Not(generate(n, size - 1))
    else:
        left = random.randint(1, size - 2)
        right = size - 1 - left
        return op([generate(n, left), generate(n, right)])


def main():
    sys.setrecursionlimit(32768)
    random.seed(flutes.__MAGIC__)
    n = 3
    proposition = "A AND NOT NOT B OR C AND NOT(A OR B)"
    with open("data/sl_satisfaction.pkl", "rb") as f:
        n, proposition = pickle.load(f)

    n = 26
    proposition = str(generate(n, 500))
    proposition = proposition.replace("and", "AND").replace("or", "OR").replace("not", "NOT")
    print(proposition)

    with flutes.work_in_progress():
        print(solution(n, proposition))
    with flutes.work_in_progress():
        print(solution_brute(n, proposition))


if __name__ == '__main__':
    main()
