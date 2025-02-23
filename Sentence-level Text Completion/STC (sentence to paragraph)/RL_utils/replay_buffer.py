import numpy as np
import math

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1)

        self.data = [None] * capacity
        self.size = 0
        self.curr_point = 0

    def add(self, data):
        self.data[self.curr_point] = data

        self.update(self.curr_point, max(self.tree[self.capacity - 1: self.capacity + self.size]) + 1)

        self.curr_point += 1
        if self.curr_point >= self.capacity:
            self.curr_point = 0

        if self.size < self.capacity:
            self.size += 1

    def update(self, point, weight):
        idx = point + self.capacity - 1
        change = weight - self.tree[idx]

        self.tree[idx] = weight

        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def get_total(self):
        return self.tree[0]

    def get_min(self):
        return min(self.tree[self.capacity - 1:self.capacity + self.size - 1])

    def sample(self, v):
        idx = 0
        while idx < self.capacity - 1:
            l_idx = idx * 2 + 1
            r_idx = l_idx + 1
            if self.tree[l_idx] >= v:
                idx = l_idx
            else:
                idx = r_idx
                v = v - self.tree[l_idx]

        point = idx - (self.capacity - 1)
        return point, self.data[point], self.tree[idx] / self.get_total()

class Memory(object):
    def __init__(self, batch_size, max_size, beta):
        self.batch_size = batch_size
        self.max_size = 2**math.floor(math.log2(max_size))
        self.beta = beta

        self._sum_tree = SumTree(max_size)

    def store_transition(self, state, action, reward, next_state, done):
        self._sum_tree.add((state, action, reward, next_state, done))

    def get_mini_batches(self):
        n_sample = self.batch_size if self._sum_tree.size >= self.batch_size else self._sum_tree.size
        total = self._sum_tree.get_total()

        step = total / n_sample
        points_transitions_probs = []
        for i in range(n_sample):
            v = np.random.uniform(i * step, (i + 1) * step)
            t = self._sum_tree.sample(v)
            points_transitions_probs.append(t)

        points, transitions, probs = zip(*points_transitions_probs)
        max_impmortance_ratio = (n_sample * self._sum_tree.get_min())**-self.beta
        importance_ratio = [(n_sample * probs[i])**-self.beta / max_impmortance_ratio
                            for i in range(len(probs))]

        return points, transitions, importance_ratio

    def update(self, points, td_error):
        for i in range(len(points)):
            self._sum_tree.update(points[i], td_error[i])