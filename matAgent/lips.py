from matAgent.baseAgent import *

# LIPS

NSIZE = 3
W = 0.5


# 相比原来的  该算法增加了对欧式距离最近的粒子的学习

class LipsSwarm(MatSwarm):
    optimizer_name = 'LIPS'
    action_space = 4

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'LIPS'

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)

        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        self.atom_history_best_x = np.zeros((self.n_part, self.n_dim))
        self.atom_history_best_fit = np.zeros(self.n_part)

        self.atom_nearest_x = np.zeros((self.n_part, NSIZE, self.n_dim))

        self.v_max = pos_max
        self.v_min = pos_min

        assert pos_max > pos_min and pos_max == -pos_min
        self.init()
        self.lips_targets = []

    def init(self):
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.fits = self.fun(self.xs)
        self.atom_history_best_fit[:] = np.inf
        self.history_best_fit = np.inf
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

        self.lips_targets.clear()

        for i in range(self.n_part):
            detax = self.xs - self.xs[i]
            distance = np.linalg.norm(detax, 2, axis=1)
            distance[distance == 0] = np.inf
            max_indexs = distance.argsort()[::-1][0:NSIZE]
            self.atom_nearest_x[i] = self.xs[max_indexs]

        for i in range(self.n_part):
            c = 1
            w = W
            if actions is not None:
                w, other_coefficient, mutation_rate = self.get_coefficients(actions, i)
                c = other_coefficient[0]
            phis = np.random.uniform(0, 4.1 / NSIZE, NSIZE)
            phi = np.sum(phis)
            pi = np.matmul(np.diag(phis), self.atom_nearest_x[i]) / NSIZE / phi
            sigema_pi = np.sum(pi, axis=0)
            self.lips_targets.append(sigema_pi)

            self.vs[i] = w * (self.vs[i] + c * phi * (sigema_pi - self.xs[i]))

        self.vs[self.vs > self.v_max] = self.v_max
        self.vs[self.vs < self.v_min] = self.v_min

        self.xs += self.vs

        self.xs[self.xs > self.pos_max] = self.pos_max
        self.xs[self.xs < self.pos_min] = self.pos_min

        self.fits = self.fun(self.xs)

    def show_method(self):
        x = self.xs[:, 0]
        y = self.xs[:, 1]
        plt.clf()
        plt.scatter(x, y, alpha=0.5)

        gbestx = self.history_best_x[0]
        gbesty = self.history_best_x[1]
        plt.scatter(gbestx, gbesty, alpha=1, edgecolors='black')

        lips_x = np.array(self.lips_targets)[:, 0]
        lips_y = np.array(self.lips_targets)[:, 1]
        plt.scatter(lips_x, lips_y, alpha=1, edgecolors='yellow')

        plt.xlim(self.pos_min, self.pos_max)
        plt.ylim(self.pos_min, self.pos_max)
        plt.pause(0.05)


def fun2(x):
    x2 = np.power(x - 50, 2)
    fit = np.sum(x2, axis=-1)
    return fit


if __name__ == '__main__':
    s = LipsSwarm(100, 40, True, fun2, 2, 100, -100, None)
    s.run()
