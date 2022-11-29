import random

import numpy as np

from matAgent.baseAgent import *

W = 0.5
C1 = 2.05
C2 = 2.05
K1 = 0.5
K2 = 0.1
# A novel particle swarm optimization based on prey–predator relationship

class PppsoSwarm(MatSwarm):
    optimizer_name = 'PPPSO'
    action_space = 5

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part * 2, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'PPPSO'
        self.pop_size_set = n_part
        self.n_part = n_part
        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)
        self.atom_best_fits = np.zeros(self.n_part * 2)
        self.active_part = [i for i in range(self.n_part)]
        self.es = []  # popsize  里面的e

        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part) + np.inf

        self.r1 = np.zeros(n_part)
        self.r2 = np.zeros(n_part)

        self.init()

    def init(self):
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.fits = self.fun(self.xs[:self.n_part])
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

        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))

        c1 = C1
        c2 = C2
        chi = 0.7298
        for i in self.active_part:
            if actions is not None:
                w, other_coefficient, mutation_rate = self.get_coefficients(actions, i)
                c1 = other_coefficient[0]
                c2 = other_coefficient[1]
                chi = w
            self.vs[i] = chi * (self.vs[i] + c1 * self.r1[i] * (self.p_best[i] - self.xs[i]) + c2 * self.r2[i] * \
                                (self.history_best_x - self.xs[i]))

            self.vs[i][self.vs[i] > self.max_v] = self.max_v
            self.vs[i][self.vs[i] < self.min_v] = self.min_v
            self.xs[i] += self.vs[i]
            self.xs[i][self.xs[i] > self.pos_max] = self.pos_max
            self.xs[i][self.xs[i] < self.pos_min] = self.pos_min
            self.fits[i] = self.fun(self.xs[i])

        gbest_index = np.argmin(self.fits)
        for i in range(self.n_part):
            if i == gbest_index:
                continue
            v = np.sum(self.vs[i] * self.vs[i])**0.5
            if np.random.random() < K1 and v < 0.01 * self.max_v:
                if np.random.random() < K2:
                    # 清除该粒子
                    if i in self.active_part:
                        self.active_part.remove(i)
                else:
                    # 粒子重设到某个pbest
                    target_pbest_index = random.choice(self.active_part)
                    self.xs[i] = self.p_best[target_pbest_index].copy()
                    self.fits[i] = self.fits[target_pbest_index]
        new_e = self.n_part - len(self.active_part)
        self.es.append(new_e)
        add_paticle_num = int(0.45 * new_e + 0.02 * (np.sum(self.es)))
        add_paticle_index = 0
        for _ in range(add_paticle_num):
            while add_paticle_index in self.active_part:
                add_paticle_index += 1
            if add_paticle_index >= 2 * self.n_part - 1:
                break
            self.active_part.append(add_paticle_index)
            self.xs[add_paticle_index] = np.random.uniform(self.pos_min, self.pos_max, self.xs[add_paticle_index].shape)
            self.vs[add_paticle_index] = np.random.uniform(self.pos_min, self.pos_max, self.xs[add_paticle_index].shape)
            self.fits[add_paticle_index] = self.fun(self.xs[add_paticle_index])
            self.p_best[add_paticle_index] = self.xs[add_paticle_index].copy()

        self.update_best()

    def show_method(self):
        x = self.xs[self.active_part, 0]
        y = self.xs[self.active_part, 1]
        plt.clf()
        plt.scatter(x, y, alpha=0.5)
        plt.xlim(self.pos_min, self.pos_max)
        plt.ylim(self.pos_min, self.pos_max)
        plt.pause(0.05)

    def get_w_c1_c2(self, actions, i):
        w, other_coefficient, mutation_rate = self.get_coefficients(actions, i)
        c1 = other_coefficient[0] * w
        c2 = other_coefficient[1] * w
        return w, c1, c2


if __name__ == '__main__':
    s = PppsoSwarm(100, 40, True, fun, 10, 1000, -1000, {})
    s.run()
    # print('best fit', test_fun(np.zeros(10)))
