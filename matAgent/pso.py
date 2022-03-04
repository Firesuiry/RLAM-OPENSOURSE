from matAgent.baseAgent import *

W = 0.5
C1 = 2
C2 = 2

class PsoSwarm(MatSwarm):
    optimizer_name = 'PSO'
    action_space = 5

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'PSO'

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)
        self.atom_best_fits = np.zeros(self.n_part)

        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        self.r1 = np.zeros(n_part)
        self.r2 = np.zeros(n_part)



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
        for i in range(self.n_part):
            if actions is not None:
                w, other_coefficient, mutation_rate = self.get_coefficients(actions, i)
                c1 = other_coefficient[0]
                c2 = other_coefficient[1]
            self.vs[i] = w * self.vs[i] + c1 * self.r1[i] * (self.p_best[i] - self.xs[i]) + c2 * self.r2[i] * \
                         (self.history_best_x - self.xs[i])

        self.vs[self.vs > self.max_v] = self.max_v
        self.vs[self.vs < self.min_v] = self.min_v
        self.xs += self.vs
        self.xs[self.xs > self.pos_max] = self.pos_max
        self.xs[self.xs < self.pos_min] = self.pos_min
        self.fits = self.fun(self.xs)

        self.update_best()


if __name__ == '__main__':
    s = PsoSwarm(100, 40, False, fun, 10, 1000, -1000, None)
    s.run()
    # print('best fit', test_fun(np.zeros(10)))
