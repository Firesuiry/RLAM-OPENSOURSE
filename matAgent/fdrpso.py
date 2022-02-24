from functions import CEC_functions
from matAgent.baseAgent import *

W = 0.5
C1 = 1
C2 = 1
C3 = 2


# FDRPSO

class FdrpsoSwarm(MatSwarm):
    optimizer_name = 'FDRPSO'
    action_space = 6

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'FDRPSO'

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)

        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        self.atom_history_best_x = np.zeros((self.n_part, self.n_dim))
        self.atom_history_best_fit = np.zeros(self.n_part)

        self.v_max = pos_max * 0.4
        self.v_min = pos_min * 0.4

        assert pos_max > pos_min and pos_max == -pos_min
        self.init()

    def init(self):

        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.fits = self.fun(self.xs)
        self.atom_history_best_fit[:] = np.inf
        self.history_best_fit = np.inf
        self.init_finish = True

    def set_x(self, x):
        assert x.shape == self.xs.shape
        assert x.shape == self.xs.shape
        self.xs = x

    def run_once(self, actions=None):
        w = W
        c1 = C1
        c2 = C2
        c3 = C3
        w = 0.9 - self.step_num / self.n_run * 0.5
        gbest_index = np.argmin(self.fits)
        if self.history_best_fit > self.fits[gbest_index]:
            self.history_best_fit = self.fits[gbest_index]
            self.history_best_x = self.xs[gbest_index].copy()
            self.best_update()

        # print(f'{self.step_num}|{w}|best fit:{self.history_best_fit}')

        for i in range(self.n_part):
            if self.atom_history_best_fit[i] > self.fits[i]:
                self.atom_history_best_fit[i] = self.fits[i]
                self.atom_history_best_x[i] = self.xs[i].copy()

        for i in range(self.n_part):
            if actions is not None:
                w, other_coefficient, mutation_rate = self.get_coefficients(actions, i)
                c1 = other_coefficient[0]
                c2 = other_coefficient[1]
                c3 = other_coefficient[2]
            fitness = self.fits[i]
            deta_fitness = self.atom_history_best_fit[i] - self.atom_history_best_fit
            for d in range(self.n_dim):
                xid = self.xs[i, d]
                distance = xid - self.atom_history_best_x[:, d]
                distance[distance == 0] = np.inf
                fdr = deta_fitness / (distance + 1e-250)
                j_index = np.argmax(fdr)
                self.vs[i, d] += w * self.vs[i, d] + \
                                 c1 * np.random.uniform(0, 1) * (self.history_best_x[d] - xid) + \
                                 c2 * np.random.uniform(0, 1) * (self.atom_history_best_x[i, d] - xid) + \
                                 c3 * np.random.uniform(0, 1) * (self.atom_history_best_x[j_index, d] - xid)
        self.vs[self.vs > self.v_max] = self.v_max
        self.vs[self.vs < self.v_min] = self.v_min

        self.xs += self.vs

        self.xs[self.xs > self.pos_max] = self.pos_max
        self.xs[self.xs < self.pos_min] = self.pos_min

        self.fits = self.fun(self.xs)

    def show_method(self):
        x = self.xs[:, 0]
        y = self.xs[:, 1]
        x2 = self.atom_history_best_x[:, 0]
        y2 = self.atom_history_best_x[:, 1]
        vx = self.vs[:, 0]
        vy = self.vs[:, 1]
        plt.clf()
        plt.scatter(x, y, alpha=0.5, edgecolors='blue')
        plt.scatter(x2, y2, alpha=0.5, edgecolors='red')
        plt.scatter(vx, vy, alpha=0.5, edgecolors='green')
        plt.xlim(self.pos_min, self.pos_max)
        plt.ylim(self.pos_min, self.pos_max)
        plt.pause(0.05)


cec_functions = CEC_functions(40)
f_num = 1

# def fun(x):
#     X = np.ones(40)
#
#     if len(x.shape) == 2:
#         ans = []
#         for row in x:
#             y = cec_functions.Y(row, f_num)
#             ans.append(y)
#     else:
#         ans = cec_functions.Y(x, f_num)
#
#     return ans


if __name__ == '__main__':
    s = FdrpsoSwarm(200, 40, True, fun, 2, 100, -100, None)
    s.run()
    print(s.result_cache)
