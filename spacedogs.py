import heapq
import random
from dataclasses import dataclass, field
from typing import List


# Credit: https://github.com/stefankoegl/kdtree
class Node(object):
    """ A Node in a kd-tree
    A tree is represented by its root node, and every node represents
    its subtree"""

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        """ Returns True if a Node has no subnodes
        >>> Node().is_leaf
        True
        >>> Node( 1, left=Node(2) ).is_leaf
        False
        """
        return (not self.data) or \
               (all(not bool(c) for c, p in self.children))

    def preorder(self):
        """ iterator for nodes: root, left, right """

        if not self:
            return

        yield self

        if self.left:
            for x in self.left.preorder():
                yield x

        if self.right:
            for x in self.right.preorder():
                yield x

    def inorder(self):
        """ iterator for nodes: left, root, right """

        if not self:
            return

        if self.left:
            for x in self.left.inorder():
                yield x

        yield self

        if self.right:
            for x in self.right.inorder():
                yield x

    def postorder(self):
        """ iterator for nodes: left, right, root """

        if not self:
            return

        if self.left:
            for x in self.left.postorder():
                yield x

        if self.right:
            for x in self.right.postorder():
                yield x

        yield self

    @property
    def children(self):
        """
        Returns an iterator for the non-empty children of the Node
        The children are returned as (Node, pos) tuples where pos is 0 for the
        left subnode and 1 for the right.
        >>> len(list(create(dimensions=2).children))
        0
        >>> len(list(create([ (1, 2) ]).children))
        0
        >>> len(list(create([ (2, 2), (2, 1), (2, 3) ]).children))
        2
        """

        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1

    def set_child(self, index, child):
        """ Sets one of the node's children
        index 0 refers to the left, 1 to the right child """

        if index == 0:
            self.left = child
        else:
            self.right = child

    def height(self):
        """
        Returns height of the (sub)tree, without considering
        empty leaf-nodes
        >>> create(dimensions=2).height()
        0
        >>> create([ (1, 2) ]).height()
        1
        >>> create([ (1, 2), (2, 3) ]).height()
        2
        """

        min_height = int(bool(self))
        return max([min_height] + [c.height() + 1 for c, p in self.children])

    def get_child_pos(self, child):
        """ Returns the position if the given child
        If the given node is the left child, 0 is returned. If its the right
        child, 1 is returned. Otherwise None """

        for c, pos in self.children:
            if child == c:
                return pos

    def __repr__(self):
        return '<%(cls)s - %(data)s>' % \
               dict(cls=self.__class__.__name__, data=repr(self.data))

    def __nonzero__(self):
        return self.data is not None

    __bool__ = __nonzero__

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data

    def __hash__(self):
        return id(self)


class KDNode(Node):
    """ A Node that contains kd-tree specific data and methods """

    def __init__(self, data=None, left=None, right=None, axis=None,
                 sel_axis=None, dimensions=None):
        """ Creates a new node for a kd-tree
        If the node will be used within a tree, the axis and the sel_axis
        function should be supplied.
        sel_axis(axis) is used when creating subnodes of the current node. It
        receives the axis of the parent node and returns the axis of the child
        node. """
        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions

    def add(self, point):
        """
        Adds a point to the current node or iteratively
        descends to one of its children.
        Users should call add() only to the topmost tree.
        """

        current = self
        while True:
            # Adding has hit an empty leaf-node, add here
            if current.data is None:
                current.data = point
                return current

            # split on self.axis, recurse either left or right
            if point[current.axis] < current.data[current.axis]:
                if current.left is None:
                    current.left = current.create_subnode(point)
                    return current.left
                else:
                    current = current.left
            else:
                if current.right is None:
                    current.right = current.create_subnode(point)
                    return current.right
                else:
                    current = current.right

    def create_subnode(self, data):
        """ Creates a subnode for the current node """

        return self.__class__(data,
                              axis=self.sel_axis(self.axis),
                              sel_axis=self.sel_axis,
                              dimensions=self.dimensions)

    def find_replacement(self):
        """ Finds a replacement for the current node
        The replacement is returned as a
        (replacement-node, replacements-parent-node) tuple """

        if self.right:
            child, parent = self.right.extreme_child(min, self.axis)
        else:
            child, parent = self.left.extreme_child(max, self.axis)

        return (child, parent if parent is not None else self)

    def should_remove(self, point, node):
        """ checks if self's point (and maybe identity) matches """
        if not self.data == point:
            return False

        return (node is None) or (node is self)

    def remove(self, point, node=None):
        """ Removes the node with the given point from the tree
        Returns the new root node of the (sub)tree.
        If there are multiple points matching "point", only one is removed. The
        optional "node" parameter is used for checking the identity, once the
        removeal candidate is decided."""

        # Recursion has reached an empty leaf node, nothing here to delete
        if not self:
            return

        # Recursion has reached the node to be deleted
        if self.should_remove(point, node):
            return self._remove(point)

        # Remove direct subnode
        if self.left and self.left.should_remove(point, node):
            self.left = self.left._remove(point)

        elif self.right and self.right.should_remove(point, node):
            self.right = self.right._remove(point)

        # Recurse to subtrees
        if point[self.axis] <= self.data[self.axis]:
            if self.left:
                self.left = self.left.remove(point, node)

        if point[self.axis] >= self.data[self.axis]:
            if self.right:
                self.right = self.right.remove(point, node)

        return self

    def _remove(self, point):
        # we have reached the node to be deleted here

        # deleting a leaf node is trivial
        if self.is_leaf:
            self.data = None
            return self

        # we have to delete a non-leaf node here

        # find a replacement for the node (will be the new subtree-root)
        root, max_p = self.find_replacement()

        # self and root swap positions
        tmp_l, tmp_r = self.left, self.right
        self.left, self.right = root.left, root.right
        root.left, root.right = tmp_l if tmp_l is not root else self, tmp_r if tmp_r is not root else self
        self.axis, root.axis = root.axis, self.axis

        # Special-case if we have not chosen a direct child as the replacement
        if max_p is not self:
            pos = max_p.get_child_pos(root)
            max_p.set_child(pos, self)
            max_p.remove(point, self)

        else:
            root.remove(point, self)

        return root

    @property
    def is_balanced(self):
        """ Returns True if the (sub)tree is balanced
        The tree is balanced if the heights of both subtrees differ at most by
        1 """

        left_height = self.left.height() if self.left else 0
        right_height = self.right.height() if self.right else 0

        if abs(left_height - right_height) > 1:
            return False

        return all(c.is_balanced for c, _ in self.children)

    def rebalance(self):
        """
        Returns the (possibly new) root of the rebalanced tree
        """

        return create([x.data for x in self.inorder()])

    def axis_dist(self, point, axis):
        """
        Squared distance at the given axis between
        the current Node and the given point
        """
        return (self.data[axis] - point[axis]) ** 2

    def dist(self, point):
        """
        Squared distance between the current Node
        and the given point
        """
        return sum((point[i] - self.data[i]) ** 2 for i in range(self.dimensions))

    def _search_one_node(self, point, result):
        if not self:
            return result

        nodeDist = self.dist(point)

        # Add current node to the priority queue if it closer than
        # at least one point in the queue.
        #
        # If the heap is at its capacity, we need to check if the
        # current node is closer than the current farthest node, and if
        # so, replace it.
        item = (-nodeDist, self)
        if result is None or -nodeDist > result[0]:
            result = item
        # get the splitting plane
        split_plane = self.data[self.axis]
        # get the squared distance between the point and the splitting plane
        # (squared since all distances are squared).
        plane_dist = point[self.axis] - split_plane
        plane_dist2 = plane_dist * plane_dist

        # Search the side of the splitting plane that the point is in
        if point[self.axis] < split_plane:
            if self.left is not None:
                result = self.left._search_one_node(point, result)
        else:
            if self.right is not None:
                result = self.right._search_one_node(point, result)

        # Search the other side of the splitting plane if it may contain
        # points closer than the farthest point in the current results.
        if result is None or -plane_dist2 > result[0]:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    result = self.right._search_one_node(point, result)
            else:
                if self.left is not None:
                    result = self.left._search_one_node(point, result)

        return result

    def search_nn(self, point):
        """
        Search the nearest node of the given point
        point must be an actual point, not a node. The nearest node to the
        point is returned. If a location of an actual node is used, the Node
        with this location will be returned (not its neighbor).
        dist is a distance function, expecting two points and returning a
        distance value. Distance values can be any comparable type.
        The result is a (node, distance) tuple.
        """

        self._point = point
        result = self._search_one_node(point, None)

        d, node = result
        return node, -d

    def is_valid(self):
        """ Checks recursively if the tree is valid
        It is valid if each node splits correctly """

        if not self:
            return True

        if self.left and self.data[self.axis] < self.left.data[self.axis]:
            return False

        if self.right and self.data[self.axis] > self.right.data[self.axis]:
            return False

        return all(c.is_valid() for c, _ in self.children) or self.is_leaf

    def extreme_child(self, sel_func, axis):
        """ Returns a child of the subtree and its parent
        The child is selected by sel_func which is either min or max
        (or a different function with similar semantics). """

        max_key = lambda child_parent: child_parent[0].data[axis]

        # we don't know our parent, so we include None
        me = [(self, None)] if self else []

        child_max = [c.extreme_child(sel_func, axis) for c, _ in self.children]
        # insert self for unknown parents
        child_max = [(c, p if p is not None else self) for c, p in child_max]

        candidates = me + child_max

        if not candidates:
            return None, None

        return sel_func(candidates, key=max_key)


def create(point_list=None, dimensions=None, axis=0, sel_axis=None):
    if point_list:
        dimensions = len(point_list[0])

    # by default cycle through the axis
    sel_axis = sel_axis or (lambda prev_axis: (prev_axis + 1) % dimensions)

    if not point_list:
        return KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)

    # Sort point list and choose median as pivot element
    point_list = list(point_list)
    point_list.sort(key=lambda point: point[axis])
    median = len(point_list) // 2

    loc = point_list[median]
    left = create(point_list[:median], dimensions, sel_axis(axis))
    right = create(point_list[median + 1:], dimensions, sel_axis(axis))
    return KDNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)


@dataclass(order=True)
class Dog:
    mass: int
    index: int
    pos: List[int] = field(compare=False)
    outdated: bool = field(compare=False, default=False)

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, item):
        return self.pos[item]


def solution(masses, locations):
    dogs = [Dog(mass, idx, pos) for idx, (mass, pos) in enumerate(zip(masses, locations))]
    heap = dogs.copy()
    heapq.heapify(heap)
    tree = create(dogs)

    cnt = 0
    while len(heap) > 1:
        cnt += 1
        dog = heapq.heappop(heap)
        while dog.outdated:
            dog = heapq.heappop(heap)
        if len(heap) == 0:
            heapq.heappush(heap, dog)
            break
        tree = tree.remove(dog)
        node, distance = tree.search_nn(dog)

        other = dogs[node.data.index]
        tree = tree.remove(other)
        other.outdated = True
        new_dog = Dog(dog.mass + other.mass, other.index, [(x + y) >> 1 for x, y in zip(dog.pos, other.pos)])
        dogs[other.index] = new_dog
        heapq.heappush(heap, new_dog)
        tree.add(new_dog)
        # if cnt % 512 == 0 and not tree.is_balanced:
        #     tree = tree.rebalance()

    return sum(heap[0].pos)


# @profile
def dist(a: Dog, b: Dog) -> int:
    return sum((x - y) ** 2 for x, y in zip(a.pos, b.pos))


def solution_truth(masses, locations):
    dogs = [Dog(mass, idx, pos) for idx, (mass, pos) in enumerate(zip(masses, locations))]
    heap = dogs.copy()
    heapq.heapify(heap)

    while len(heap) > 1:
        dog = heapq.heappop(heap)
        while dog.outdated:
            dog = heapq.heappop(heap)
        if len(heap) == 0:
            heapq.heappush(heap, dog)
            break

        min_dist, best_idx = min((dist(dog, other), idx) for idx, other in enumerate(heap) if not other.outdated)
        other = heap[best_idx]
        other.outdated = True
        new_dog = Dog(dog.mass + other.mass, other.index, [(x + y) // 2 for x, y in zip(dog.pos, other.pos)])
        dogs[other.index] = new_dog
        heapq.heappush(heap, new_dog)

    return sum(heap[0].pos)


import flutes


def main():
    random.seed(19260817)
    with flutes.work_in_progress():
        n = 5000
        k = 5
        masses = [random.randint(0, 10000) for _ in range(n)]
        locations = [tuple(random.randint(-10000, 10000) for _ in range(k)) for _ in range(n)]
    # with open("/Users/kanari/Downloads/sl_spacedogs.pkl", "rb") as f:
    #     masses, locations = pickle.load(f)
    # masses = [2, 5, 4]
    # locations = [(1, 4), (3, 1), (2, 6)]
    with flutes.work_in_progress():
        print(solution(masses, locations))
    with flutes.work_in_progress():
        print(solution_truth(masses, locations))


if __name__ == '__main__':
    main()
