from matAgent.baseAgent import *

# OLPSO

# 采用正交化选择

G = 5  # 多少轮不改善重新OED
C = 2  # 速度更新倍乘参数


class OlpsoSwarm(MatSwarm):
    optimizer_name = 'OLPSO'
    action_space = 4

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'OLPSO'

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)

        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)
        self.g_best_index = 0

        self.atom_history_best_x = np.zeros((self.n_part, self.n_dim))
        self.atom_history_best_fit = np.zeros(self.n_part)

        self.atom_stagnated = np.zeros(self.n_part)
        self.atom_orthogonal_target = np.zeros((self.n_part, self.n_dim), dtype=np.int)

        self.target_X = np.zeros((self.n_part, self.n_dim))
        self.orthogonal_target_display = np.zeros((self.n_part, self.n_dim))
        self.orthogonal_target_display[:, :] = 20

        self.v_max = pos_max
        self.v_min = pos_min

        self.oa = self.get_OA()

        assert pos_max > pos_min and pos_max == -pos_min
        self.init()

    def init(self):
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.v_min, self.v_max, self.xs.shape)
        self.fits = self.fun(self.xs)
        self.atom_history_best_fit = self.fits.copy()
        self.atom_history_best_x = self.xs.copy()
        self.g_best_index = best_index = np.argmin(self.fits)
        self.history_best_fit = self.fits[best_index]
        self.history_best_x = self.xs[best_index].copy()

        for i in range(self.n_part):
            self.orthogonal(i)
        self.init_finish = True

    def set_x(self, x):
        assert x.shape == self.xs.shape
        self.xs = x

    def orthogonal(self, particle_index):
        self.atom_stagnated[particle_index] = 0
        target_index = self.g_best_index
        if particle_index == self.g_best_index:
            target_index = np.random.randint(0, self.n_part)

        xs = [self.atom_history_best_x[particle_index], self.atom_history_best_x[target_index]]
        test_x = np.zeros(self.n_dim)
        test_num = self.oa.shape[0]
        res = np.zeros(self.n_dim, dtype=np.int)
        test_ress = np.zeros(test_num)
        # 进行测试
        for i in range(test_num):
            test_vec = self.oa[i]
            for d in range(self.n_dim):
                test_x[d] = xs[test_vec[d]][d]
            fit = self.fun(test_x)
            test_ress[i] = fit

        # 对每个维度计算平均值
        dim_fit = np.zeros((self.n_dim, 2))
        for d in range(self.n_dim):
            for i in range(test_num):
                dim_fit[d, self.oa[i, d]] += test_ress[i]
            # print('dim:{} dim_fit:{}'.format(d, dim_fit))
            best_index = np.argmin(dim_fit[d])
            res[d] = best_index

        # 对最终结果进行处理
        for d in range(self.n_dim):
            if res[d] == 0:
                res[d] = particle_index
            elif res[d] == 1:
                res[d] = target_index

        # target dispaly
        target = np.zeros(self.n_dim)
        for d in range(self.n_dim):
            target[d] = self.atom_history_best_x[res[d]][d]

        # print('orthogonal :{}'.format(particle_index))
        self.orthogonal_target_display[particle_index] = target
        # print('id:{} input1:{}\nid:{} input2:{}'.format(particle_index, self.atom_history_best_x[particle_index],
        #                                                 target_index, self.atom_history_best_x[target_index]))
        # print('output:{}'.format(target))
        # print('res :{} gbest:{} particle_index:{}'.format(res, self.g_best_index, particle_index))
        self.atom_orthogonal_target[particle_index] = res

    def get_OA(self):
        # the method is from OLPSO paper appendix
        # step 1
        m = np.int(np.power(2, np.ceil(np.log2(self.n_dim + 1))))
        n = m - 1
        u = np.int(np.ceil(np.log2(self.n_dim + 1)))
        # print(m, n, u)
        oa = np.zeros((m, n), dtype=np.int)
        k = 1
        # step 2
        for a in range(1, m + 1, 1):
            for k in range(1, u + 1, 1):
                b = np.power(2, k - 1)
                oa[a - 1, b - 1] = np.floor((a - 1) / np.power(2, u - k)) % 2

        # step 3
        for b in range(1, n + 1, 1):
            for a in range(1, m + 1, 1):
                if np.floor(np.log2(b)) == np.log2(b):
                    continue
                else:
                    ori_b = np.int(np.power(2, np.floor(np.log2(b))))
                    s = b - ori_b
                    oa[a - 1, b - 1] = (oa[a - 1][s - 1] + oa[a - 1][ori_b - 1]) % 2
        # step 4
        # oa = oa + 1
        return oa

    def show_method(self):
        x = self.xs[:, 0]
        y = self.xs[:, 1]
        plt.clf()
        plt.scatter(x, y, alpha=0.5)
        x2 = self.target_X[:, 0]
        y2 = self.target_X[:, 1]
        plt.scatter(x2, y2, alpha=0.5, edgecolors='red')
        # x2 = self.atom_history_best_x[:, 0]
        # y2 = self.atom_history_best_x[:, 1]
        # plt.scatter(x2, y2, alpha=0.5, edgecolors='yellow')
        x2 = self.orthogonal_target_display[:, 0]
        y2 = self.orthogonal_target_display[:, 1]
        plt.scatter(x2, y2, alpha=0.5, edgecolors='green')

        plt.xlim(self.pos_min, self.pos_max)
        plt.ylim(self.pos_min, self.pos_max)
        plt.pause(0.05)

    def run_once(self, actions=None):
        iter_num = self.step_num + 1
        w = 0.9 - iter_num / self.n_run * 0.5
        c = C
        # print('best fit:{}'.format(self.history_best_fit))

        for i in range(self.n_part):
            if actions is not None:
                w, other_coefficient, mutation_rate = self.get_coefficients(actions, i)
                c = other_coefficient[0]
            for d in range(self.n_dim):
                # if self.atom_orthogonal_target[i, d]:
                #     pod = self.atom_history_best_x[self.g_best_index, d]
                # else:
                #     pod = self.atom_history_best_x[i, d]
                pod = self.atom_history_best_x[self.atom_orthogonal_target[i, d], d]
                # print('target :{}'.format(pod))
                self.vs[i, d] = w * self.vs[i, d] + c * np.random.uniform(0, 1) * (pod - self.xs[i, d])
                if self.vs[i, d] > self.v_max:
                    self.vs[i, d] = self.v_max
                if self.vs[i, d] < self.v_min:
                    self.vs[i, d] = self.v_min
                self.target_X[i, d] = pod

        self.vs[self.vs > self.v_max] = self.v_max
        self.vs[self.vs < self.v_min] = self.v_min

        self.xs += self.vs

        self.xs[self.xs > self.pos_max] = self.pos_max
        self.xs[self.xs < self.pos_min] = self.pos_min

        self.fits = self.fun(self.xs)

        self.g_best_index = gbest_index = np.argmin(self.fits)
        if self.history_best_fit > self.fits[gbest_index]:
            self.history_best_fit = self.fits[gbest_index]
            self.history_best_x = self.xs[gbest_index].copy()
            self.best_update()

        for i in range(self.n_part):
            if self.atom_history_best_fit[i] > self.fits[i]:
                # print('ori fit:{} new fit:{} orix:{} newx:{}'.format(self.atom_history_best_fit[i], self.fits[i],
                #                                                      self.atom_history_best_x[i], self.xs[i]))
                # print('orifit:{} newfit:{} detax:{}'.format(self.fun(self.atom_history_best_x[i]), self.fun(self.xs[i]),
                #                                             self.atom_history_best_x[i] - self.xs[i]))
                self.atom_history_best_fit[i] = self.fits[i].copy()
                self.atom_history_best_x[i] = self.xs[i].copy()
            else:
                self.atom_stagnated[i] += 1
                if self.atom_stagnated[i] >= G:
                    self.orthogonal(i)
        pass


if __name__ == '__main__':
    s = OlpsoSwarm(100, 10, False, fun, 40, 100, -100, None)
    s.run()
    # input()
    # print(s.get_OA())
