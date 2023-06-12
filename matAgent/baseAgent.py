from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from env.NormalEnv import NormalEnv
from rl.DDPG.TF2_DDPG_Basic import DDPG
from train.ddpg import get_ddpg_object


class MatSwarm:
    optimizer_name = 'base_optimizer'
    action_space = 0
    obs_space = 3 * 5

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        if show:
            print('群初始化 参数：{}'.format(locals()))
        self.n_run = n_run
        self.n_part = n_part
        self.show = show
        self.fitness_valuate = fun
        self.config = config_dic

        self.step_num = 0

        self.atoms = []
        self.n_group = config_dic.get('group', 1)
        self.n_dim = n_dim
        self.pos_max = pos_max
        self.pos_min = pos_min
        self.max_v = 0.2 * (pos_max - pos_min)
        self.min_v = -self.max_v

        self.xs = np.zeros([n_part, n_dim])
        self.fits = np.zeros(n_part, )

        self.history_best_x = np.zeros(self.n_dim)
        self.history_best_fit = np.inf
        self.p_best = np.zeros_like(self.xs)
        self.atom_history_best_fits = np.zeros(self.n_part) + np.inf

        self.fe_num = 0
        if config_dic is None:
            config_dic = {}
        self.fe_max = config_dic.get('max_fes', 10000)
        self.run_flag = True
        self.record_per_fe = 1000
        self.result_cache = []
        self.init_finish = False

        self.ddpg_actor = None
        model = config_dic.get('model')
        if model:
            model_name = Path(model).name
            self.name = f'{self.optimizer_name}-{model_name}'
            self.optimizer_name = self.name

            gym_env = NormalEnv(show=False, obs_shape=(self.obs_space,),
                                action_shape=(self.action_space * self.n_group,))
            ddpg = get_ddpg_object(gym_env, )
            print(ddpg.actor.summary())
            ddpg.load_actor(str(model))

            self.ddpg_actor = ddpg

        self.last_best_update_fe = 0

    def show_method(self):
        x = self.xs[:, 0]
        y = self.xs[:, 1]
        plt.clf()
        plt.scatter(x, y, alpha=0.5)
        plt.xlim(self.pos_min, self.pos_max)
        plt.ylim(self.pos_min, self.pos_max)
        plt.pause(0.05)

    def data_collect_method(self):
        fits = self.fits
        mean = np.mean(fits)
        std = np.std(fits)
        best = self.history_best_fit
        self.result_cache.append((self.fe_num, mean, best, std))

    def run(self, *args, **kwargs):
        self.step_num = 0
        while self.step_num < self.n_run:
            self.step_num += 1
            if self.run_flag:
                actions = None
                if self.ddpg_actor:
                    state = self.get_state()
                    actions = self.ddpg_actor.policy(state).numpy()
                if self.show:
                    print('群运行中：{}'.format(self.step_num))
                self.run_once(actions=actions)
            else:
                print('fe 超过限制 当前fe:{} 最大fe:{}'.format(self.fe_num, self.fe_max))
                break
            if self.show:
                self.show_method()

    def run_once(self, actions=None):
        pass

    def update_info(self, *args):
        pass

    def fun(self, x):
        if len(x.shape) == 2:
            ans = []
            for row in x:
                y = self.fitness_valuate(np.float64(row))
                self.add_check_fe()
                ans.append(y)
        else:
            ans = self.fitness_valuate(np.float64(x))
            self.add_check_fe()
        return np.array(ans)

    def add_check_fe(self):
        if self.init_finish:
            if (self.fe_num % self.record_per_fe == 0 or self.fe_num == self.fe_max) and self.fe_num <= self.fe_max:
                self.data_collect_method()
            if self.fe_num >= self.fe_max:
                # self.data_collect_method()
                self.run_flag = False
            self.fe_num += 1

    def get_coefficients(self, actions, i, coefficients_multi=True, range_process=True):
        group = int(len(actions) / self.action_space)
        action = actions[i % group * self.action_space:i % group * self.action_space + self.action_space]
        multi_coefficient = action[-1] + 1
        mutation_rate = (action[-2] + 1) * 0.01
        if range_process:
            other_coefficient = action[1:-2] * 1.5 + 1.5
            w = action[0] * 0.4 + 0.5
        else:
            other_coefficient = action[1:-2]
            w = action[0]
        action_sum = np.sum(other_coefficient) + 1e-10
        if action_sum == 0:
            action_sum = 1e-10
        if coefficients_multi:
            other_coefficient = other_coefficient / action_sum * multi_coefficient * 4
        return w, other_coefficient, mutation_rate

    def get_group_coefficients(self, actions, i):
        group = int(len(actions) / self.action_space)
        action = actions[i % group * self.action_space:i % group * self.action_space + self.action_space]
        return action.copy()

    def get_state(self):
        process = self.fe_num * 2 / self.fe_max - 1
        no_improve_fe = (self.fe_num - self.last_best_update_fe) / self.fe_max
        diversity = np.mean(np.std(self.xs, axis=0))
        next_state = [process, no_improve_fe, diversity]

        return sin_encode(next_state, num=4)

    def best_update(self):
        self.last_best_update_fe = self.fe_num


def sin_encode(state, num=3):
    new_state = []
    for s in state:
        new_state.append(s)
        for i in range(num):
            # new_state.append(np.sin(s * 2 ** i))
            new_state.append(s * 2 ** i)
    return np.sin(new_state)


def fun(x):
    x2 = np.power(x, 2)
    fit = np.sum(x2, axis=-1)
    return fit


if __name__ == '__main__':
    pass
