from matAgent.baseAgent import *


W = 0.5
C1 = 2
C2 = 2

CLPSO_M = 7

NSIZE = 3


class HrlepsoBaseSwarm(MatSwarm):
    optimizer_name = 'HRLEPSO'
    action_space = 6

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'H-RELPSO-BASE'

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)
        self.atom_history_best_fits = np.zeros(self.n_part)

        self.max_v = 0.2 * (pos_max - pos_min)
        self.min_v = -self.max_v

        self.g_best = np.zeros(n_dim)
        self.g_best_index = 0
        self.fits = np.zeros(self.n_part)

        self.r1 = np.zeros(n_part)
        self.r2 = np.zeros(n_part)

        # CLPSO target
        self.atom_pci = np.zeros(self.n_part)
        indexs = np.array(list(range(self.n_part)))
        self.atom_pci = 0.05 + 0.45 * np.exp(10 * indexs / (self.n_part - 1)) / (np.exp(10) - 1)
        self.flag = np.zeros(self.n_part)  # 与OLPSO共用
        self.fid = np.zeros((self.n_part, self.n_dim), dtype=np.int)

        self.atom_nearest_x = np.zeros((self.n_part, NSIZE, self.n_dim))

        # OLPSO
        self.oa = self.get_OA()
        self.atom_orthogonal_target = np.zeros((self.n_part, self.n_dim), dtype=np.int)

        self.old_data = {}
        self.old_data['mean'] = 0
        self.old_data['best'] = 0

        self.fit_value = [0, 0, 0, 0, 0]

        self.lips_targets = []

        self.init()

    def init(self):
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.max_v, self.min_v, self.xs.shape)
        self.fits = self.fun(self.xs)
        self.init_finish = True

        self.gbest_index = np.argmin(self.fits)
        self.history_best_fit = self.fits[self.gbest_index]
        self.g_best = self.xs[self.gbest_index].copy()
        self.atom_history_best_fits = self.fits.copy()
        self.p_best = self.xs.copy()

        for i in range(self.n_part):
            self.caculate_fid(i)
            # self.orthogonal(i)

    def set_x(self, x):
        assert x.shape == self.xs.shape
        self.xs = x

    def caculate_fid(self, i):
        #  Selection of exemplar dimensions for particle i.
        rands = np.random.uniform(0, 1, self.n_dim)
        for d in range(self.n_dim):
            rand = rands[d]
            if rand < self.atom_pci[i]:
                fids = np.random.randint(0, self.n_part, 2)
                fits = self.atom_history_best_fits[fids]
                self.fid[i, d] = fids[np.argmin(fits)]
            else:
                self.fid[i, d] = i

    def update_best(self):
        for i in range(self.n_part):
            # print(self.fits.shape,self.atom_best_fits.shape)
            if self.fits[i] < self.atom_history_best_fits[i]:
                self.p_best[i] = self.xs[i].copy()
                self.atom_history_best_fits[i] = self.fits[i]
                self.flag[i] = 0
            else:
                if self.flag[i] > CLPSO_M:
                    self.caculate_fid(i)
                    # self.orthogonal(i)
                    self.flag[i] = 0
                else:
                    self.flag[i] += 1

        self.gbest_index = np.argmin(self.fits)
        if self.history_best_fit > self.fits[self.gbest_index]:
            # print(f'update best to {self.fits[self.gbest_index]}')
            self.history_best_fit = self.fits[self.gbest_index]
            self.g_best = self.xs[self.gbest_index].copy()
            self.best_update()

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

    def orthogonal(self, particle_index):
        target_index = self.g_best_index
        if particle_index == self.g_best_index:
            target_index = np.random.randint(0, self.n_part)

        xs = [self.p_best[particle_index], self.p_best[target_index]]
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
        # target = np.zeros(self.n_dim)
        # for d in range(self.n_dim):
        #     target[d] = self.p_best[res[d]][d]

        # print('orthogonal :{}'.format(particle_index))
        # self.orthogonal_target_display[particle_index] = target
        # print('id:{} input1:{}\nid:{} input2:{}'.format(particle_index, self.p_best[particle_index],
        #                                                 target_index, self.p_best[target_index]))
        # print('output:{}'.format(target))
        # print('res :{} gbest:{} particle_index:{}'.format(res, self.g_best_index, particle_index))
        self.atom_orthogonal_target[particle_index] = res

    # r0 分群 r1 clpso r2 fdr r3 lips r4 oed r5 gbest r6 pbest r7 速度惯性 R8 没增长突变 R9 V=0突变概率
    def run_once(self, actions=np.zeros(50)):

        if self.show:
            print('{}|best fit:{}'.format(self.fe_num / self.fe_max, self.history_best_fit))

        self.lips_targets.clear()
        for i in range(self.n_part):
            w, other_coefficient, mutation_rate = self.get_coefficients(actions, i)
            # 单粒子迭代
            fdr_deta_fitness = self.atom_history_best_fits[i] - self.atom_history_best_fits
            # 将粒子分为5大群 分别进行处理

            # w = action[7] * 0.4 + 0.5
            r1 = other_coefficient[0]
            # r2 = action[2] * 1.5 + 1.5
            r2 = 0
            # r3 = action[3] * 1.5 + 1.5
            r3 = 0
            # r4 = action[4] * 1.5 + 1.5
            r4 = 0
            r5 = other_coefficient[1]
            r6 = other_coefficient[2]


            # LIPS
            # ________________________________________________________________
            # for ii in range(self.n_part):
            #     detax = self.xs - self.xs[ii]
            #     distance = np.linalg.norm(detax, 2, axis=1)
            #     distance[distance == 0] = np.inf
            #     max_indexs = distance.argsort()[::-1][0:NSIZE]
            #     self.atom_nearest_x[ii] = self.xs[max_indexs]
            # phis = np.random.uniform(0, 4.1 / NSIZE, NSIZE)
            # phi = np.sum(phis)
            # pi = np.matmul(np.diag(phis), self.atom_nearest_x[i]) / NSIZE / phi
            # lips_targets = sigema_pi = np.sum(pi, axis=0)
            # self.lips_targets.append(lips_targets)
            # ________________________________________________________________
            for d in range(self.n_dim):
                # 维度

                # LIPS
                # lips_target = lips_targets[d]

                # CLPSO
                clpso_target = self.p_best[self.fid[i, d], d]

                # FDR
                # xid = self.xs[i, d]
                # distance = xid - self.p_best[:, d]
                # distance[i] = np.inf
                # fdr = fdr_deta_fitness / (distance + 1e-250)
                # j_index = np.argmax(fdr)
                # fdr_target = self.p_best[j_index, d]

                # OLPSO
                # olpso_target = self.p_best[self.atom_orthogonal_target[i, d], d]

                location = self.xs[i, d]
                v_delta_v = w * self.vs[i, d]
                clpso_delta_v = r1 * np.random.uniform(0, 1) * (clpso_target - location)
                # fdr_delta_v = r2 * np.random.uniform(0, 1) * (fdr_target - location)
                # lips_delta_v = r3 * phi * np.random.uniform(0, 1) * (lips_target - location)
                # olpso_delta_v = r4 * np.random.uniform(0, 1) * (olpso_target - location)
                gbest_delta_v = r5 * np.random.uniform(0, 1) * (self.g_best[d] - location)
                pbest_delta_v = r6 * np.random.uniform(0, 1) * (self.p_best[i, d] - location)
                new_v = v_delta_v + clpso_delta_v + gbest_delta_v + pbest_delta_v
                self.vs[i, d] = new_v
                # if new_v != self.vs[i, d]:
                #     print(f'v{self.vs[i, d]} to newV{new_v}')

            if np.random.random() < mutation_rate * self.flag[i]:
                self.xs[i] = np.random.uniform(self.pos_min, self.pos_max, self.xs[i].shape)

        self.vs[self.vs > self.max_v] = self.max_v
        self.vs[self.vs < self.min_v] = self.min_v
        # for i in range(len(self.vs)):
        #     if np.linalg.norm(self.vs[i]) > self.max_v:
        #         self.vs[i] = self.max_v * self.vs[i] / np.linalg.norm(self.vs[i])
        self.xs += self.vs
        self.xs[self.xs > self.pos_max] = self.pos_max
        self.xs[self.xs < self.pos_min] = self.pos_min
        self.fits = self.fun(self.xs)

        self.update_best()
        # if self.step_num == self.n_run and self.fe_num < self.fe_max:
        #     self.step_num -= 1

    def show_method(self):

        x = self.xs[:, 0]
        y = self.xs[:, 1]
        x2 = self.p_best[:, 0]
        y2 = self.p_best[:, 1]
        vx = self.vs[:, 0]
        vy = self.vs[:, 1]

        plt.clf()
        plt.scatter(x, y, alpha=0.5, edgecolors='blue')
        plt.scatter(x2, y2, alpha=0.5, edgecolors='red')
        plt.scatter(vx, vy, alpha=0.5, edgecolors='green')
        gbestx = self.g_best[0]
        gbesty = self.g_best[1]
        # print(self.g_best)
        plt.scatter(gbestx, gbesty, alpha=1, edgecolors='black')
        # lips_x = np.array(self.lips_targets)[:, 0]
        # lips_y = np.array(self.lips_targets)[:, 1]
        # plt.scatter(lips_x, lips_y, alpha=1, edgecolors='yellow')
        plt.xlim(self.pos_min, self.pos_max)
        plt.ylim(self.pos_min, self.pos_max)
        plt.pause(0.05)

    # def get_state(self):
    #
    #     # mean_fit = np.mean(self.fits)
    #     # old_mean = self.old_data['mean']
    #     # old_best = self.old_data['best']
    #     # self.old_data['mean'] = mean_fit
    #     # deta_mean = mean_fit - old_mean
    #     #
    #     # self.fit_value.append(deta_mean)
    #     # del self.fit_value[0]
    #     #
    #     # next_state = self.fit_value.copy()
    #     # next_state.append(self.fe_num / self.fe_max - 0.5)
    #
    #     next_state = [(self.fe_num / self.fe_max - 0.5) * 2, ]
    #
    #     return next_state
        # return np.array([self.fe_num / self.fe_max])


def fun2(x):
    x2 = np.power(x - 50, 2)
    fit = np.sum(x2, axis=-1)
    return fit


def fit(x):
    return np.sum(np.power(x, 2))


if __name__ == '__main__':
    s = TestpsoSwarm(100, 40, True, fit, 2, 100, -100, {'max_fes': 100000})
    s.run()
    # print('best fit', test_fun(np.zeros(10)))
