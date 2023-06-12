from matAgent.baseAgent import *

W = 0.5
C1 = 2
C2 = 2

# HPSO-TVAC

# 时间变分层次化粒子群算法

# c1 c2的变化参数
C1F = 0.5
C1I = 2.5
C2F = 2.5
C2I = 0.5

W_0 = 0.9
W_1 = 0.4


# 速度重新初始化参数


class HpsotvacSwarm(MatSwarm):
    optimizer_name = 'HPSO-TVAC'
    action_space = 5

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'HPSO-TVAC'

        # self.xs = np.zeros([n_part, n_dim])
        # self.fits = np.zeros(n_part, )

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)

        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        self.r1 = np.zeros(n_part)
        self.r2 = np.zeros(n_part)

        assert pos_max > pos_min
        assert pos_max == -pos_min

        self.last_time_best_fit = np.inf

        self.init()

    def init(self):

        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.fits = self.fun(self.xs)
        self.init_finish = True

    def set_x(self, x):
        assert x.shape == self.xs.shape
        self.xs = x

    def update_best(self):
        for i in range(self.n_part):
            # print(self.fits.shape,self.atom_best_fits.shape)
            if self.fits[i] < self.atom_history_best_fits[i]:
                self.p_best[i] = self.xs[i].copy()
                self.atom_history_best_fits[i] = self.fits[i]

            if self.history_best_fit > self.fits[i]:
                # print(f'update best to {self.fits[self.gbest_index]}')
                self.history_best_fit = self.fits[i]
                self.g_best = self.xs[i].copy()
                self.best_update()
    def get_w_c1_c2(self, actions, i):
        w_coefficient, other_coefficient, mutation_rate = self.get_coefficients(actions, i,
                                                                                coefficients_multi=False,
                                                                                range_process=False)
        w = w_coefficient * 0.2 + W_0 + (W_1 - W_0) * self.fe_num / self.fe_max
        c1 = other_coefficient[0] * 0.2 + (C1F - C1I) * self.fe_num / self.fe_max + C1I
        c2 = other_coefficient[1] * 0.2 + (C2F - C2I) * self.fe_num / self.fe_max + C2I
        return w, c1, c2

    def run_once(self, actions=None):
        temp_c1 = c1 = (C1F - C1I) * self.fe_num / self.fe_max + C1I
        temp_c2 = c2 = (C2F - C2I) * self.fe_num / self.fe_max + C2I
        temp_w = w = W_0 + (W_1 - W_0) * self.fe_num / self.fe_max

        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        for i in range(self.n_part):
            if actions is not None:
                w_coefficient, other_coefficient, mutation_rate = self.get_coefficients(actions, i,
                                                                                        coefficients_multi=False,
                                                                                        range_process=False)
                temp_w = w_coefficient * 0.2 + w
                temp_c1 = other_coefficient[0] * 0.2 + c1
                temp_c2 = other_coefficient[1] * 0.2 + c2
            self.vs[i] = temp_w * self.vs[i] + \
                         temp_c1 * self.r1[i] * (self.p_best[i] - self.xs[i]) + \
                         temp_c2 * self.r2[i] * (self.g_best - self.xs[i])

        for i in range(self.vs.shape[0]):
            for d in range(self.vs.shape[1]):
                if self.vs[i, d] == 0:
                    self.vs[i, d] = np.random.uniform(self.min_v, self.max_v)
        self.vs[self.vs > self.max_v] = self.max_v
        self.vs[self.vs < self.min_v] = self.min_v

        self.xs += self.vs
        self.xs[self.xs > self.pos_max] = self.pos_max
        self.xs[self.xs < self.pos_min] = self.pos_min

        self.fits = self.fun(self.xs)
        self.update_best()


if __name__ == '__main__':
    s = HpsotvacSwarm(100000, 100, True, fun, 2, 10, -10, config_dic={})
    s.run()
