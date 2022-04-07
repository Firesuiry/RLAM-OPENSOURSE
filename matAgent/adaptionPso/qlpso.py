import numpy as np

from matAgent.baseAgent import *

W = 0.5
C1 = 2
C2 = 2


class QlpsoSwarm(MatSwarm):
    optimizer_name = 'QLPSO'
    action_space = 5

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'QLPSO'

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)
        self.atom_best_fits = np.zeros(self.n_part)

        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        self.r1 = np.zeros(n_part)
        self.r2 = np.zeros(n_part)

        self.q_table = np.zeros((4, 4, 4))

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

    def run_once(self, *args, **kwargs):
        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))

        w = 0.9 - 0.7 * self.step_num / self.n_run
        c1 = C1
        c2 = C2
        deta_R = self.pos_max - self.pos_min
        fit_min = np.min(self.fits)
        deta_F = np.max(self.fits) - fit_min
        actions = [
            [1, 2.5, 0.5],
            [0.8, 2, 1],
            [0.6, 1, 2],
            [0.4, 0.5, 2.5],
        ]
        for i in range(self.n_part):
            action_index = 0
            distance_index = 0
            fit_deta_index = 0
            if self.fe_num / self.fe_max < 0.9:
                distance = np.sqrt(np.sum((self.xs[i] - self.history_best_x) ** 2))
                fit_deta = self.history_best_fit - self.fits[i]
                while distance_index / 4 * deta_R < distance:
                    distance_index += 1
                while fit_deta_index / 4 * deta_F < (self.fits[i] - fit_min):
                    fit_deta_index += 1
                distance_index -= 1
                fit_deta_index -= 1
                distance_index = min(distance_index, 3)
                fit_deta_index = min(fit_deta_index, 3)
                action_index = np.argmax(self.q_table[distance_index, fit_deta_index])
                w, c1, c2 = actions[action_index]
            else:
                w, c1, c2 = (0, 0, 3)
            self.vs[i] = w * self.vs[i] + c1 * self.r1[i] * (self.p_best[i] - self.xs[i]) + c2 * self.r2[i] * \
                         (self.history_best_x - self.xs[i])
            self.vs[i][self.vs[i] > self.max_v] = self.max_v
            self.vs[i][self.vs[i] < self.min_v] = self.min_v
            self.xs[i] += self.vs[i]
            self.xs[i][self.xs[i] > self.pos_max] = self.pos_max
            self.xs[i][self.xs[i] < self.pos_min] = self.pos_min
            last_fit = self.fits[i]
            self.fits[i] = self.fun(self.xs[i])
            if self.fe_num / self.fe_max < 0.9:
                reward = self.get_reward(action_index, self.fits[i] < last_fit)
                alpha = 0.1
                gamma = 0.9
                self.q_table[distance_index, fit_deta_index, action_index] = \
                    (1 - alpha) * self.q_table[distance_index, fit_deta_index, action_index] + \
                    alpha * (reward + gamma * np.max(self.q_table[distance_index, fit_deta_index]))
        self.update_best()

    def get_reward(self, action, imporve):
        process = self.fe_num / self.fe_max
        if imporve:
            if action == 0:
                return 4 - process * 3
            elif action == 1:
                return 3 - process * 1
            elif action == 2:
                return 2 + process * 1
            else:
                return 1 + process * 3
        else:
            if action == 0:
                return -1 - process * 3
            elif action == 1:
                return -2 - process * 1
            elif action == 2:
                return -3 + process * 1
            else:
                return -4 + process * 3



if __name__ == '__main__':
    s = QlpsoSwarm(1000, 100, True, fun, 10, 1000, -1000, {})
    s.run()
    # print('best fit', test_fun(np.zeros(10)))
