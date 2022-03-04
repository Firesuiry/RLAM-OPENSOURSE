import numpy as np

from matAgent.baseAgent import *
from pyit2fls import *

W = 0.5
C1 = 2
C2 = 2


class FT2PsoSwarm(MatSwarm):
    optimizer_name = 'FT2PSO'
    action_space = 5

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'FT2PSO'

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

        # iter
        iteration = self.fe_num / self.fe_max

        # diversity
        diversity = np.mean(np.std(self.xs - self.history_best_x, axis=0))

        min_diversity = np.inf
        max_diversity = -np.inf
        for i in range(self.n_part):
            single_diversity = np.std(self.xs[i] - self.history_best_x)
            if single_diversity < min_diversity:
                min_diversity = single_diversity
            if single_diversity > max_diversity:
                max_diversity = single_diversity
        if min_diversity == max_diversity:
            normal_diversity = 0
        else:
            normal_diversity = (diversity - min_diversity) / (max_diversity - min_diversity)

        # error
        mean_fit = np.mean(self.fits)
        min_fit = np.min(self.fits)
        max_fit = np.max(self.fits)
        normal_fit = (mean_fit - min_fit) / (max_fit - min_fit) if max_fit != min_fit else 1

        c1, c2 = get_c1c2(iteration=iteration, diversity=normal_diversity, error=normal_fit)

        for i in range(self.n_part):
            # if actions is not None:
            #     w, other_coefficient, mutation_rate = self.get_coefficients(actions, i)
            #     c1 = other_coefficient[0]
            #     c2 = other_coefficient[1]
            self.vs[i] = w * self.vs[i] + c1 * self.r1[i] * (self.p_best[i] - self.xs[i]) + c2 * self.r2[i] * \
                         (self.history_best_x - self.xs[i])

        self.vs[self.vs > self.max_v] = self.max_v
        self.vs[self.vs < self.min_v] = self.min_v
        self.xs += self.vs
        self.xs[self.xs > self.pos_max] = self.pos_max
        self.xs[self.xs < self.pos_min] = self.pos_min
        self.fits = self.fun(self.xs)

        self.update_best()


fuzzy_system = None
domain = np.linspace(-1, 4, 501)


def get_it2fs(left, center, right, height):
    斜率 = height / (center - left)
    x_offset = 0.4 / 斜率
    return IT2FS(domain, tri_mf, [left - x_offset, center, right + x_offset, height + 0.4], tri_mf,
                 [left + x_offset, center, right - x_offset, height - 0.4])


def init_fuzzy_system():
    global fuzzy_system, domain
    input_low = IT2FS(domain, tri_mf, [-0.2, 0.5, 1.2, 1.4], tri_mf, [0.2, 0.5, 0.8, 0.6])
    input_medium = get_it2fs(0, 0.5, 1, 1)
    input_high = get_it2fs(0.5, 1, 1.5, 1)

    output_low = get_it2fs(0, 0.5, 1, 1)
    output_medium_low = get_it2fs(0.5, 1, 1.5, 1)
    output_medium = get_it2fs(1, 1.5, 2, 1)
    output_medium_high = get_it2fs(1.5, 2, 2.5, 1)
    output_high = get_it2fs(2, 2.5, 3, 1)

    SYS = IT2FLS()
    SYS.add_input_variable("iteration")
    SYS.add_input_variable("diversity")
    SYS.add_input_variable("error")
    SYS.add_output_variable("C1")
    SYS.add_output_variable("C2")
    # 1-9
    SYS.add_rule([("iteration", input_low), ("diversity", input_low), ("error", input_low)],
                 [("C1", output_medium_low), ("C2", output_low)])
    SYS.add_rule([("iteration", input_low), ("diversity", input_low), ("error", input_medium)],
                 [("C1", output_medium_high), ("C2", output_low)])
    SYS.add_rule([("iteration", input_low), ("diversity", input_low), ("error", input_high)],
                 [("C1", output_high), ("C2", output_low)])

    SYS.add_rule([("iteration", input_low), ("diversity", input_medium), ("error", input_low)],
                 [("C1", output_medium), ("C2", output_medium)])
    SYS.add_rule([("iteration", input_low), ("diversity", input_medium), ("error", input_medium)],
                 [("C1", output_medium_high), ("C2", output_medium_low)])
    SYS.add_rule([("iteration", input_low), ("diversity", input_medium), ("error", input_high)],
                 [("C1", output_medium_high), ("C2", output_low)])

    SYS.add_rule([("iteration", input_low), ("diversity", input_high), ("error", input_low)],
                 [("C1", output_low), ("C2", output_medium_low)])
    SYS.add_rule([("iteration", input_low), ("diversity", input_high), ("error", input_medium)],
                 [("C1", output_medium), ("C2", output_medium)])
    SYS.add_rule([("iteration", input_low), ("diversity", input_high), ("error", input_high)],
                 [("C1", output_medium_low), ("C2", output_low)])

    # 10-18
    SYS.add_rule([("iteration", input_medium), ("diversity", input_low), ("error", input_low)],
                 [("C1", output_medium), ("C2", output_medium)])
    SYS.add_rule([("iteration", input_medium), ("diversity", input_low), ("error", input_medium)],
                 [("C1", output_medium_high), ("C2", output_medium_low)])
    SYS.add_rule([("iteration", input_medium), ("diversity", input_low), ("error", input_high)],
                 [("C1", output_medium_high), ("C2", output_low)])

    SYS.add_rule([("iteration", input_medium), ("diversity", input_medium), ("error", input_low)],
                 [("C1", output_medium_low), ("C2", output_medium_high)])
    SYS.add_rule([("iteration", input_medium), ("diversity", input_medium), ("error", input_medium)],
                 [("C1", output_medium), ("C2", output_medium)])
    SYS.add_rule([("iteration", input_medium), ("diversity", input_medium), ("error", input_high)],
                 [("C1", output_medium_high), ("C2", output_medium_low)])

    SYS.add_rule([("iteration", input_medium), ("diversity", input_high), ("error", input_low)],
                 [("C1", output_low), ("C2", output_medium_high)])
    SYS.add_rule([("iteration", input_medium), ("diversity", input_high), ("error", input_medium)],
                 [("C1", output_medium_low), ("C2", output_medium_high)])
    SYS.add_rule([("iteration", input_medium), ("diversity", input_high), ("error", input_high)],
                 [("C1", output_medium), ("C2", output_medium)])

    # 19-27
    SYS.add_rule([("iteration", input_high), ("diversity", input_low), ("error", input_low)],
                 [("C1", output_low), ("C2", output_medium_low)])
    SYS.add_rule([("iteration", input_high), ("diversity", input_low), ("error", input_medium)],
                 [("C1", output_medium), ("C2", output_medium)])
    SYS.add_rule([("iteration", input_high), ("diversity", input_low), ("error", input_high)],
                 [("C1", output_medium_low), ("C2", output_low)])

    SYS.add_rule([("iteration", input_high), ("diversity", input_medium), ("error", input_low)],
                 [("C1", output_low), ("C2", output_medium_high)])
    SYS.add_rule([("iteration", input_high), ("diversity", input_medium), ("error", input_medium)],
                 [("C1", output_medium_low), ("C2", output_medium_high)])
    SYS.add_rule([("iteration", input_high), ("diversity", input_medium), ("error", input_high)],
                 [("C1", output_medium), ("C2", output_medium)])

    SYS.add_rule([("iteration", input_high), ("diversity", input_high), ("error", input_low)],
                 [("C1", output_low), ("C2", output_high)])
    SYS.add_rule([("iteration", input_high), ("diversity", input_high), ("error", input_medium)],
                 [("C1", output_low), ("C2", output_medium_high)])
    SYS.add_rule([("iteration", input_high), ("diversity", input_high), ("error", input_high)],
                 [("C1", output_low), ("C2", output_medium_low)])

    fuzzy_system = SYS


def get_c1c2(iteration, diversity, error):
    global fuzzy_system, domain
    if fuzzy_system == None:
        init_fuzzy_system()
    # print(iteration, diversity, error)
    s, c = fuzzy_system.evaluate({"iteration": iteration, "diversity": diversity, "error": error}, min_t_norm,
                                 max_s_norm, domain,
                                 method="Centroid", algorithm="KM")
    return (crisp(c['C1']), crisp(c['C2']))


if __name__ == '__main__':
    s = FT2PsoSwarm(10000, 40, True, fun, 10, 1000, -1000, {})
    s.run()
    # print('best fit', test_fun(np.zeros(10)))
    # init_fuzzy_system()
    # print(get_c1c2(0.5, 0.5, 0.5))
