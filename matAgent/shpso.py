from matAgent.baseAgent import *

NSIZE = 3
W = 0.72
C1 = 2
C2 = 2


# sHPSO

# 预定义了多种搜索方法，提前定义每个粒子使用的方法


class ShpsoSwarm(MatSwarm):
    optimizer_name = 'SHPSO'
    action_space = 5

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'SHPSO'

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)

        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        self.atom_history_best_x = np.zeros((self.n_part, self.n_dim))
        self.atom_history_best_fit = np.zeros(self.n_part)

        self.v_max = pos_max
        self.v_min = pos_min

        self.atom_method = np.zeros(self.n_part)

        assert pos_max > pos_min and pos_max == -pos_min
        self.init()

    def init(self):
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.fits = self.fun(self.xs)
        self.atom_history_best_fit[:] = np.inf
        self.history_best_fit = np.inf

        self.atom_method = np.random.randint(0, 5, self.n_part)
        self.init_finish = True

    def set_x(self, x):
        assert x.shape == self.xs.shape
        self.xs = x

    def run_once(self, actions=None):
        gbest_index = np.argmin(self.fits)
        if self.history_best_fit > self.fits[gbest_index]:
            self.history_best_fit = self.fits[gbest_index]
            self.history_best_x = self.xs[gbest_index].copy()
            self.best_update()

        # print('best fit:{}'.format(self.history_best_fit))

        for i in range(self.n_part):
            if self.atom_history_best_fit[i] > self.fits[i]:
                self.atom_history_best_fit[i] = self.fits[i]
                self.atom_history_best_x[i] = self.xs[i].copy()

        for i in range(self.n_part):
            method = self.atom_method[i]
            w = W
            c1 = 2.5 - 2 * (self.step_num + 1) / self.n_run
            c2 = 0.5 + 2 * (self.step_num + 1) / self.n_run
            if actions is not None:
                group_id = self.atom_method[i]
                w, other_coefficient, mutation_rate = self.get_coefficients(actions, group_id, coefficients_multi=False)
                c1 = (other_coefficient[0] - 1.5) * 0.2 + c1
                c2 = (other_coefficient[1] - 1.5) * 0.2 + c2
            # 0 标准PSO
            # 1 只跟随自身最优
            # 2 只跟随全局最优
            # 3 直接更新xs 目标是自身最优和全局最优的中点的高斯分布
            # 4 概率更新xs 目标是自身最优或3的点
            if method < 3:
                if method == 0:
                    self.vs[i] = w * self.vs[i] + \
                                 c1 * np.random.uniform(0, 1) * (self.atom_history_best_x[i] - self.xs[i]) \
                                 + c2 * np.random.uniform(0, 1) * (self.history_best_x - self.xs[i])
                elif method == 1:
                    self.vs[i] = w * self.vs[i] + \
                                 c1 * np.random.uniform(0, 1) * (self.atom_history_best_x[i] - self.xs[i])
                elif method == 2:
                    self.vs[i] = w * self.vs[i] + \
                                 + c2 * np.random.uniform(0, 1) * (self.history_best_x - self.xs[i])
                self.vs[i][self.vs[i] > self.v_max] = self.v_max
                self.vs[i][self.vs[i] < self.v_min] = self.v_min
                self.xs[i] += self.vs[i]
            else:
                if method == 3:
                    for d in range(self.n_dim):
                        self.xs[i, d] = np.random.normal(
                            (self.atom_history_best_x[i, d] + self.history_best_x[d]) / 2,
                            np.abs(self.atom_history_best_x[i, d] - self.history_best_x[d]))
                if method == 4:
                    for d in range(self.n_dim):
                        if np.random.uniform(0, 1) < 0.5:
                            self.xs[i, d] = np.random.normal(
                                (self.atom_history_best_x[i, d] + self.history_best_x[d]) / 2,
                                np.abs(self.atom_history_best_x[i, d] - self.history_best_x[d]))
                        else:
                            self.xs[i, d] = self.atom_history_best_x[i, d]

        self.xs[self.xs > self.pos_max] = self.pos_max
        self.xs[self.xs < self.pos_min] = self.pos_min

        self.fits = self.fun(self.xs)


if __name__ == '__main__':
    s = ShpsoSwarm(100, 10, True, fun, 2, 10, -10, None)
    s.run()
