import numpy as np

from matAgent.baseAgent import *

W = 0.5
C1 = 2
C2 = 2


class GwoSwarm(MatSwarm):
    optimizer_name = 'GWO'
    action_space = 6

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'GWO'

        self.alpha_pos = np.zeros(n_dim)
        self.alpha_fit = np.inf
        self.beta_pos = np.zeros(n_dim)
        self.beta_fit = np.inf
        self.delta_pos = np.zeros(n_dim)
        self.delta_fit = np.inf

        self.init()

    def init(self):
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.fits = self.fun(self.xs)

        self.init_finish = True

    def update_best(self):
        for i in range(self.n_part):
            if self.fits[i] < self.alpha_fit:
                self.alpha_pos = self.xs[i].copy()
                self.alpha_fit = self.fits[i]
                self.history_best_fit = self.alpha_fit
                self.best_update()
            elif self.fits[i] < self.beta_fit:
                self.beta_pos = self.xs[i].copy()
                self.beta_fit = self.fits[i]
            elif self.fits[i] < self.delta_fit:
                self.delta_pos = self.xs[i].copy()
                self.delta_fit = self.fits[i]

    def run_once(self, actions=None):
        a_parameter = 2 - 2 * self.fe_num / self.fe_max
        a_parameter1 = a_parameter2 = a_parameter3 = a_parameter
        c1m = c2m = c3m = 1
        # print('{}|best fit:{}'.format(self.step_num / self.n_run, self.history_best_fit))
        for i in range(self.n_part):
            if actions is not None:
                action = self.get_group_coefficients(actions, i)
                mutli_coefficients = 0.3
                a_parameter1 = a_parameter + mutli_coefficients * action[0]
                a_parameter2 = a_parameter + mutli_coefficients * action[1]
                a_parameter3 = a_parameter + mutli_coefficients * action[2]
                c1m = action[3]*mutli_coefficients + 1
                c2m = action[4]*mutli_coefficients + 1
                c3m = action[5]*mutli_coefficients + 1

            for j in range(self.n_dim):
                r_1 = np.random.random()  # r_1 is a random number in [0,1]
                r_2 = np.random.random()  # r_2 is a random number in [0,1]

                a_1 = 2 * a_parameter1 * r_1 - a_parameter1  # Equation (3.3)
                c_1 = 2 * r_2 * c1m  # Equation (3.4)

                d_alpha = abs(c_1 * self.alpha_pos[j] - self.xs[i][j])  # Equation (3.5)-part 1
                x_1 = self.alpha_pos[j] - a_1 * d_alpha  # Equation (3.6)-part 1

                r_1 = np.random.random()
                r_2 = np.random.random()

                a_2 = 2 * a_parameter2 * r_1 - a_parameter2  # Equation (3.3)
                c_2 = 2 * r_2 * c2m  # Equation (3.4)

                d_beta = abs(c_2 * self.beta_pos[j] - self.xs[i][j])  # Equation (3.5)-part 2
                x_2 = self.beta_pos[j] - a_2 * d_beta  # Equation (3.6)-part 2

                r_1 = np.random.random()
                r_2 = np.random.random()

                a_3 = 2 * a_parameter3 * r_1 - a_parameter3  # Equation (3.3)
                c_3 = 2 * r_2 * c3m  # Equation (3.4)

                d_delta = abs(c_3 * self.delta_pos[j] - self.xs[i][j])  # Equation (3.5)-part 3
                x_3 = self.delta_pos[j] - a_3 * d_delta  # Equation (3.5)-part 3

                self.xs[i][j] = (x_1 + x_2 + x_3) / 3

        self.xs[self.xs > self.pos_max] = self.pos_max
        self.xs[self.xs < self.pos_min] = self.pos_min
        self.fits = self.fun(self.xs)
        self.update_best()


if __name__ == '__main__':
    s = GwoSwarm(100, 40, True, fun, 10, 1000, -1000, {})
    s.run()
    # print('best fit', test_fun(np.zeros(10)))
