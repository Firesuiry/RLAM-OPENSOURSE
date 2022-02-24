from env.EnvBase import Env
import numpy as np
from agent.pso import PsoSwarm
from target_function import *
import random


def fit(x):
    return -np.sum(np.power(x, 2))


def sqrt(a, n):
    if a == 0:
        return 0
    b = -1 * (a < 0) + 1 * (a > 0)
    return (a * b) ** (1 / n) * b


class PsoEnv(Env):
    def __init__(self, show=False):
        super().__init__(obs_shape=(6,), action_shape=(3,), action_low=-1, action_high=1)
        self.pso_swarm = None
        self.fit_value = [0., 0., 0., 0., 0.]

        self.step_num = 0
        self.show_flag = show

        self.run_time = 0

    def reset(self):
        """

        :return: next_state
        """
        n_dim = 100
        self.n_run = n_run = 100
        n_part = 40
        show = self.show_flag
        pos_max = np.array((10, 10))
        pos_min = np.array((-10, -10))
        n_group = 1
        targetSwarm = PsoSwarm
        func = random.choice(benchmarks)
        func = benchmarks[0]
        self.fun = func()
        print('训练函数:{}'.format(func.__name__))
        self.pso_swarm = targetSwarm(n_run, n_part, show, self.fun.fun, n_dim, self.fun.max, self.fun.min)
        self.pso_swarm.init_swarm()

        self.fit_value = [0., 0., 0., 0., 0.]
        self.step_num = 0
        self.old_data = {
            'mean': 0,
            'best': 0,
        }

        next_state, reword, done, _ = self.step(None, init=True)
        return next_state

    def test(self):
        done = False
        step_num = 0
        self.reset()
        while not done:
            a, b, done, c = self.step(None, True)
            step_num += 1
        return step_num

    def step(self, action, init=False):
        """
        :param action: 动作
        :return:
        next_state
        reword
        done
        none
        """
        if init:
            pass
        else:
            action = action.numpy()
        done = False
        self.step_num += 1

        if self.pso_swarm.show:
            self.pso_swarm.show_method()
        self.pso_swarm.before_explore()
        for atom in self.pso_swarm.atoms:
            if init:
                atom.explore()
            else:
                atom.explore(action[0] / 2.5 + 0.5, action[1] + 1.5, action[2] + 1.5)
        self.pso_swarm.after_explore()

        if self.pso_swarm.best_fit < self.fun.finish or self.step_num >= self.n_run:
            done = True

        num = 0
        fit = 0
        for atom in self.pso_swarm.atoms:
            num += 1
            fit += atom.fitness()
        mean_fit = fit / num
        old_mean = self.old_data['mean']
        old_best = self.old_data['best']
        self.old_data['mean'] = mean_fit
        self.old_data['best'] = self.pso_swarm.best_fit

        deta_best = self.pso_swarm.best_fit - old_best
        deta_mean = mean_fit - old_mean

        if not init:
            self.fit_value.append(deta_mean)
            del self.fit_value[0]

        next_state = self.fit_value.copy()
        next_state.append(self.step_num * 0.001)

        if deta_best < 0:
            # reward = sqrt(deta_best, 3)
            reward = 1
        else:
            reward = -1

        if np.isnan(reward):
            print(deta_best)
            exit()

        if init:
            reward = 0
        if self.show_flag:
            print('action:{} next_state:{} reward:{} done:{} best:{}'.format(action, next_state, reward, done,
                                                                             self.pso_swarm.best_fit))
        if done:
            if self.show_flag:
                print('迭代次数：{}\n'.format(self.step_num))
            with open('res2.json', 'a', encoding='utf-8') as f:
                f.write('迭代次数：{}\n'.format(self.step_num))
        return np.array(next_state), reward, done, None
