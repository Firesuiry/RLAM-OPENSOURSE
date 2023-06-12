from env.EnvBase import Env
import numpy as np

from functions import CEC_functions
from matAgent.hrlepso_base import HrlepsoBaseSwarm

import random

a = '''
-266.0531289
135544184.9
1.82774E+11
76113.16795
-549.3809779
-594.32925
-515.8652937
-678.834181
-537.7759724
904.025237
11.72616567
496.6955526
492.4799812
8093.931058
13883.27105
203.7485402
965.8900094
1331.777365
1512.912731
624.2726595
2518.154593
10873.44775
15514.33257
1378.749903
1493.122829
1651.331007
3327.011517
5873.175731
'''

mins = a.split()
mins = [float(a) for a in mins]


# mins = [-1231.212332, -204.6357241, 5.775145931, 141.0081423, 347.7206203, 6719.742783, 11518.91779, 203.2561533, 911.3600662, 1008.23017, 617.5997075, 108875421.5, 622.6670525, 1582.465063, 9477.469788, 14676.14278, 1364.45416, 1492.538031, 1507.858461, 3325.571622, 2030.524316, 37696395997.0, 72681.67153, -839.4878675, -662.095693, -678.2112235, -678.8561225, -540.4918971]


def fit(x):
    return np.sum(np.power(x, 2))


def sqrt(a, n):
    if a == 0:
        return 0
    b = -1 * (a < 0) + 1 * (a > 0)
    return (a * b) ** (1 / n) * b


class function_wrapper:
    max = 10
    min = -10
    finish = 1e-15

    def __init__(self, dim, fun_num):
        self.cec_functions = CEC_functions(dim)
        self.fun_num = fun_num

    def fun(self, x):
        if len(x.shape) == 2:
            ans = []
            for row in x:
                y = self.cec_functions.Y(row, self.fun_num)
                ans.append(y)
        else:
            ans = self.cec_functions.Y(x, self.fun_num)

        return np.array(ans)


class HrlepsoEnv(Env):
    def __init__(self, show=False):
        super().__init__(obs_shape=(1,), action_shape=(50,), action_low=-1, action_high=1)
        self.pso_swarm = None
        self.fit_value = [0., 0., 0., 0., 0.]

        self.step_num = 0
        self.show_flag = show

        self.fun_num = -1
        self.min_value = 0

        self.run_time = 0

    def reset(self):
        """

        :return: next_state
        """
        n_dim = 50
        self.n_run = n_run = 1000
        n_part = 40
        show = self.show_flag
        pos_max = np.array((10, 10))
        pos_min = np.array((-10, -10))
        n_group = 1
        targetSwarm = HrlepsoBaseSwarm
        fun_num = random.randint(1, 28)
        self.fun_num = fun_num
        self.min_value = mins[fun_num - 1]
        print(f'初始化测试函数：{fun_num},最小值:{self.min_value}')

        fun_class = function_wrapper(50, fun_num)
        # self.pso_swarm = targetSwarm(n_run, n_part, show, fun_class.fun, n_dim, 100, -100, {'max_fes': 20000})
        self.pso_swarm = targetSwarm(n_run, n_part, show, fun_class.fun, n_dim, 100, -100, {'max_fes': 20000})
        # self.min_value = 1e-17

        self.fit_value = [0., 0., 0., 0., 0.]
        self.step_num = 0
        self.old_data = {
            'mean': 0,
            'best': 0,
        }

        # next_state, reword, done, _ = self.step(None, init=True)
        return self.pso_swarm.get_state()

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

        action = action.numpy()
        done = False
        self.step_num += 1

        if self.pso_swarm.show:
            self.pso_swarm.show_method()

        self.pso_swarm.run_once(action)

        # if self.pso_swarm.best_fit < self.fun.finish or self.step_num >= self.n_run:
        #     done = True
        if not self.pso_swarm.run_flag:
            done = True

        num = 0
        # fit = 0
        # for atom in self.pso_swarm.atoms:
        #     num += 1
        #     fit += atom.fitness()
        # mean_fit = fit / num
        # old_mean = self.old_data['mean']
        old_best = self.old_data['best']
        # self.old_data['mean'] = mean_fit
        self.old_data['best'] = self.pso_swarm.history_best_fit

        deta_best = self.pso_swarm.history_best_fit - old_best
        # deta_mean = mean_fit - old_mean

        # if not init:
        #     self.fit_value.append(deta_mean)
        #     del self.fit_value[0]

        next_state = self.pso_swarm.get_state()
        # next_state.append(self.step_num * 0.001)
        # print(f'state:{next_state}\naction:{action[:10]}\nmean:{np.mean(action)},std:{np.std(action)}')

        if deta_best < 0:
            # reward = sqrt(deta_best, 3)
            reward = 1
        else:
            reward = -1

        if np.isnan(reward):
            print(deta_best)
            exit()

        if self.pso_swarm.history_best_fit < self.min_value:
            done = True
            reward = 10

        if init:
            reward = 0
        if self.show_flag:
            print('action:{} next_state:{} reward:{} done:{} best:{}'.format(action, next_state, reward, done,
                                                                             self.pso_swarm.history_best_fit))
        if done:
            res = f'迭代次数：{self.step_num},测试函数:{self.fun_num}，函数目标值：{self.min_value},运行结果：{self.pso_swarm.history_best_fit}'
            print(res)
            if self.show_flag:
                print('迭代次数：{}'.format(self.step_num))
            with open('res2.json', 'a', encoding='utf-8') as f:
                f.write(f'{res}\n')
        return np.array(next_state), reward, done, None
