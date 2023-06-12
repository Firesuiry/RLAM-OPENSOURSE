import random

import numpy as np

from matAgent.baseAgent import *
from pyit2fls import *

W = 0.5
C1 = 2
C2 = 2
H = 20


class SuccessHistoryPsoSwarm(MatSwarm):
    optimizer_name = 'SuccessHistoryPso'
    action_space = 5

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'SuccessHistoryPso'

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)
        self.atom_best_fits = np.zeros(self.n_part)

        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        self.r1 = np.zeros(n_part)
        self.r2 = np.zeros(n_part)

        self.history_memory = np.ones((2, H)) * 0.5
        self.history_memory_index = 0

        self.init()

    def init(self):
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.fits = self.fun(self.xs)
        self.init_finish = True

        gbest_index = np.argmin(self.fits)
        self.history_best_fit = self.fits[gbest_index]
        self.history_best_x = self.xs[gbest_index].copy()
        self.atom_best_fits = self.fits.copy()
        self.p_best = self.xs.copy()

    def set_x(self, x):
        assert x.shape == self.xs.shape
        self.xs = x

    def update_best(self):
        for i in range(self.n_part):
            # print(self.fits.shape,self.atom_best_fits.shape)
            if self.fits[i] < self.atom_best_fits[i]:
                self.p_best[i] = self.xs[i].copy()
                self.atom_best_fits[i] = self.fits[i]

        gbest_index = np.argmin(self.fits)
        if self.history_best_fit > self.fits[gbest_index]:
            self.history_best_fit = self.fits[gbest_index]
            self.history_best_x = self.xs[gbest_index].copy()
            self.best_update()

    def run_once(self, actions=None):
        w = 0.9 - 0.7 * self.step_num / self.n_run

        # print('{}|best fit:{}'.format(self.step_num / self.n_run, self.history_best_fit))

        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        w = W
        c1 = C1
        c2 = C2
        better_c1_list = []
        better_c2_list = []
        for i in range(self.n_part):
            c1_base = np.random.normal(random.choice(self.history_memory[0]), 0.1)
            c1_base = np.min((np.max((0, c1_base)), 1))
            c2_base = random.choice(self.history_memory[1])
            c2_base = np.min((np.max((0, c2_base)), 1))
            c1 = c1_base * 2.5
            c2 = c2_base * 2.5
            self.vs[i] = w * self.vs[i] + c1 * self.r1[i] * (self.p_best[i] - self.xs[i]) + c2 * self.r2[i] * \
                         (self.history_best_x - self.xs[i])
            self.vs[i][self.vs[i] > self.max_v] = self.max_v
            self.vs[i][self.vs[i] < self.min_v] = self.min_v
            self.xs[i] += self.vs[i]
            self.xs[i][self.xs[i] > self.pos_max] = self.pos_max
            self.xs[i][self.xs[i] < self.pos_min] = self.pos_min
            new_fit = self.fun(self.xs[i])
            if new_fit < self.fits[i]:
                self.fits[i] = new_fit
                better_c1_list.append(c1_base)
                better_c2_list.append(c2_base)

        if len(better_c1_list) > 0:
            self.history_memory[0][self.history_memory_index] = np.mean(better_c1_list)
            self.history_memory[1][self.history_memory_index] = np.mean(better_c2_list)
            self.history_memory_index = (self.history_memory_index + 1) % H

        self.update_best()


if __name__ == '__main__':
    s = SuccessHistoryPsoSwarm(10000, 40, True, fun, 10, 1000, -1000, {})
    s.run()
    # print('best fit', test_fun(np.zeros(10)))
    # init_fuzzy_system()
    # print(get_c1c2(0.5, 0.5, 0.5))
