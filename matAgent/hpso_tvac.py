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

        self.v_max = pos_max
        self.v_min = pos_min
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

    def run_once(self, actions=None):
        c1 = (C1F - C1I) * (self.step_num + 1) / self.n_run + C1I
        c2 = (C2F - C2I) * (self.step_num + 1) / self.n_run + C2I
        w = W_0 + (W_1 - W_0) * (self.step_num + 1) / self.n_run
        gbest_index = np.argmin(self.fits)

        new_global_best_fit = self.fits[gbest_index].copy()
        # deta_global = new_global_best_fit - self.last_time_best_fit
        self.last_time_best_fit = new_global_best_fit

        if self.history_best_fit > self.fits[gbest_index]:
            self.history_best_fit = self.fits[gbest_index].copy()
            self.history_best_x = self.xs[gbest_index].copy()
            self.best_update()

        # print('best fit:{}'.format(self.history_best_fit))

        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        for i in range(self.n_part):
            if actions is not None:
                w, other_coefficient, mutation_rate = self.get_coefficients(actions, i)
                c1 = other_coefficient[0]
                c2 = other_coefficient[1]
            self.vs[i] = w * self.vs[i] + c1 * self.r1[i] * (self.p_best[i] - self.xs[i]) + c2 * self.r2[i] * \
                         (self.history_best_x - self.xs[i])

        for i in range(self.vs.shape[0]):
            for d in range(self.vs.shape[1]):
                if self.vs[i, d] == 0:
                    self.vs[i, d] = np.random.uniform(self.v_min, self.v_max)
        self.vs[self.vs > self.v_max] = self.v_max
        self.vs[self.vs < self.v_min] = self.v_min

        self.xs += self.vs

        self.fits = self.fun(self.xs)


if __name__ == '__main__':
    s = HpsotvacSwarm(100, 10, True, fun, 2, 10, -10, config_dic={})
    s.run()
