from matAgent.baseAgent import *

W = 0.5
C1 = 2
C2 = 2

# EPSO 共5种算法
NUM_STRATEGY = 5

MIN_VALUE = 0.01
LP = 50

# 多次没优化阈值
M = 5


class EpsoSwarm(MatSwarm):
    optimizer_name = 'EPSO'


    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'EPSO'

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)

        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        self.r1 = np.zeros(n_part)
        self.r2 = np.zeros(n_part)

        self.n_part1 = 8
        self.n_part2 = self.n_part - self.n_part1

        self.max_v = 0.2 * (pos_max - pos_min)
        self.min_v = -self.max_v

        self.fid = np.zeros((self.n_part, self.n_dim), dtype=np.int)
        self.atom_history_best_x = np.zeros((self.n_part, self.n_dim))
        self.atom_history_best_fit = np.zeros(self.n_part)

        # EPSO 一些策略参数
        self.pk = np.ones(NUM_STRATEGY) / NUM_STRATEGY
        self.rk = np.array(range(NUM_STRATEGY)) / NUM_STRATEGY
        self.success_mem = np.zeros(NUM_STRATEGY)
        self.failure_mem = np.zeros(NUM_STRATEGY)
        self.sk = np.zeros(NUM_STRATEGY)
        # ----------------

        max_iter_list = np.array(list(range(1, self.n_run + 10, 1)))
        max_iteration = self.n_run
        # CLPSO
        self.c1 = 3 - max_iter_list * 1.5 / self.n_run
        self.w1 = 0.9 - max_iter_list * 0.5 / self.n_run
        # Sa-pso
        self.w2 = 0.9 - max_iter_list * (0.7 / self.n_run)
        # PSO
        self.c2_1 = 2.5 - max_iter_list * 2 / self.n_run
        self.c2_2 = 0.5 + max_iter_list * 2 / self.n_run
        # Method 2: FDR_PSO
        self.fii = [1, 1, 2]
        # Method3: HPSO_TVAC
        self.c3_1 = 2.5 - max_iter_list * 2 / max_iteration
        self.c3_2 = 0.5 + max_iter_list * 2 / max_iteration
        self.re_init_vel = pos_max - max_iter_list * (0.9 * pos_max) / max_iteration

        # Method4: LIPS
        self.nsize = 3

        # Method 5: CLPSO
        self.c4_1 = 2.5 - max_iter_list * 2 / max_iteration
        self.c4_2 = 0.5 + max_iter_list * 2 / max_iteration
        self.clpso_flag = np.zeros(self.n_part)
        self.fri_best = np.array(range(1, self.n_part, 1))
        indexs = np.array(list(range(self.n_part)))
        self.atom_pci = 0.05 + 0.45 * np.exp(10 * indexs / (self.n_part - 1)) / (np.exp(10) - 1)

        self.init()

        for i in range(self.n_part):
            if i < self.n_part1:
                self.caculate_fid(i, self.n_part1)
            else:
                self.caculate_fid(i)

    def caculate_fid(self, i, particle_num=0):
        #  Selection of exemplar dimensions for particle i.
        if particle_num == 0:
            particle_num = self.n_part
        rands = np.random.uniform(0, 1, self.n_dim)
        for d in range(self.n_dim):
            rand = rands[d]
            if rand < self.atom_pci[i]:
                fids = np.random.randint(0, particle_num, 2)
                fits = self.atom_history_best_fit[fids]
                self.fid[i, d] = fids[np.argmin(fits)]
            else:
                self.fid[i, d] = i

    def init(self):
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.pos_min * 0.5, self.pos_max * 0.5, self.xs.shape)
        self.fits = self.fun(self.xs)
        self.p_best = self.xs.copy()
        best_index = np.argmin(self.fits)
        self.g_best = self.xs[best_index]
        self.atom_history_best_x = self.p_best
        self.atom_history_best_fit = self.fits.copy()
        self.history_best_x = self.g_best

        self.init_finish = True

    def set_x(self, x):
        assert x.shape == self.xs.shape
        self.xs = x

    def run_once(self, actions=None):
        # print('iter:{} best:{}'.format(self.step_num, self.history_best_fit))
        if self.fe_num < 0.9 * self.fe_max:
            self.op1_clpso()
            self.op2_epso(range(self.n_part1, self.n_part, 1))
        else:
            self.op2_epso(range(0, self.n_part, 1))
        if self.fe_num < self.fe_max and self.step_num == self.n_run:
            print('当前fe:{} 设定fe:{} 增加迭代'.format(self.fe_num, self.fe_max))
            self.step_num -= 1

    def op1_clpso(self):
        # 前几个粒子进行CLPSO操作
        for i in range(self.n_part1):
            if self.clpso_flag[i] > M:
                self.caculate_fid(i, self.n_part1)
            for d in range(self.n_dim):
                self.update_paricle(i, d)
            self.fits[i] = self.fun(self.xs[i])
            x = self.xs[i]
            if (x > self.pos_min).all() and (x < self.pos_max).all():
                # 表示在范围内
                if self.fits[i] < self.atom_history_best_fit[i]:
                    # 找到新的最优值
                    # print('找到新的最优值')
                    self.atom_history_best_fit[i] = self.fits[i]
                    self.atom_history_best_x[i] = self.xs[i]
                    self.clpso_flag[i] = 0
                    if self.fits[i] < self.history_best_fit:
                        self.history_best_fit = self.fits[i]
                        self.history_best_x = self.xs[i].copy()
                        self.best_update()
                else:
                    self.clpso_flag[i] += 1

    def op2_epso(self, op_parts):
        for i in op_parts:
            if self.step_num > 1:
                total = self.success_mem + self.failure_mem
                total[total == 0] = 1
                self.sk = self.success_mem / total + MIN_VALUE
                self.pk = self.sk / np.sum(self.sk)
                self.rk = np.cumsum(self.pk)
                self.success_mem[:] = 0
                self.failure_mem[:] = 0
            prob = np.random.uniform(0, 1)
            if prob < self.rk[0]:  # 标准PSO
                strategy_k = 0
                delta = self.c2_1[self.step_num] * np.random.uniform(0, 1, self.n_dim) * (
                        self.atom_history_best_x[i] - self.xs[i]) + \
                        self.c2_2[self.step_num] * np.random.uniform(0, 1, self.n_dim) * (
                                self.history_best_x - self.xs[i])
                self.vs[i] = self.w2[self.step_num] * self.vs[i]
            elif prob < self.rk[1]:  # FDR-PSO
                strategy_k = 1
                fitness = self.fits[i]
                deta_fitness = fitness - self.history_best_fit
                for d in range(self.n_dim):
                    xid = self.xs[i, d]
                    distance = xid - self.atom_history_best_x[:, d]
                    fdr = deta_fitness / (distance + 1e-250)
                    j_index = np.argmax(fdr)
                    self.vs[i, d] += self.w2[self.step_num] * self.vs[i, d] + \
                                     self.fii[0] * np.random.uniform(0, 1) * (self.history_best_x[d] - xid) + \
                                     self.fii[1] * np.random.uniform(0, 1) * (self.atom_history_best_x[i, d] - xid) + \
                                     self.fii[2] * np.random.uniform(0, 1) * (
                                             self.atom_history_best_x[j_index, d] - xid)
            elif prob < self.rk[2]:  # HPSO
                strategy_k = 2
                r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
                r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))

                self.vs = self.c3_1[self.step_num] * r1 * (self.p_best - self.xs) + self.c3_2[self.step_num] * r2 * (
                        self.history_best_x - self.xs)

                for d in range(self.vs.shape[1]):
                    if self.vs[i, d] == 0:
                        self.vs[i, d] = np.random.uniform(-self.re_init_vel[self.step_num],
                                                          self.re_init_vel[self.step_num])
                    if self.vs[i, d] > 100:
                        self.vs[i, d] = 100
                    elif self.vs[i, d] < -100:
                        self.vs[i, d] = -100
            elif prob < self.rk[3]:  # LIPS
                strategy_k = 3
                detax = self.xs - self.xs[i]
                distance = np.linalg.norm(detax, 2, axis=1)
                distance[distance == 0] = np.inf
                max_indexs = distance.argsort()[::-1][0:self.nsize]
                atom_nearest_xs = self.xs[max_indexs]

                phis = np.random.uniform(0, 4.1 / self.nsize, self.nsize)
                phi = np.sum(phis)
                pi = np.matmul(np.diag(phis), atom_nearest_xs) / self.nsize / phi
                sigema_pi = np.sum(pi, axis=0)
                self.vs[i] = W * (self.vs[i] + phi * (sigema_pi - self.xs[i]))

            else:  # CLPSO
                strategy_k = 4
                if self.clpso_flag[i] > M:
                    self.caculate_fid(i, self.n_part1)
                for d in range(self.n_dim):
                    self.update_paricle(i, d)
                self.fits[i] = self.fun(self.xs[i])
                x = self.xs[i]

            self.vs[i][self.vs[i] > self.max_v] = self.max_v
            self.vs[i][self.vs[i] < self.min_v] = self.min_v

            self.xs[i] += self.vs[i]
            x = self.xs[i]
            if (x > self.pos_min).all() and (x < self.pos_max).all():
                self.fits[i] = self.fun(self.xs[i])
                if self.fits[i] < self.atom_history_best_fit[i]:
                    self.atom_history_best_fit[i] = self.fits[i]
                    self.atom_history_best_x[i] = self.xs[i]
                    self.success_mem[strategy_k] += 1
                    if strategy_k == 4:  # 是CLPSO
                        self.clpso_flag[i] = 0
                else:
                    self.failure_mem[strategy_k] += 1
                    if strategy_k == 4:  # 是CLPSO
                        self.clpso_flag[i] += 1

                if self.fits[i] < self.history_best_fit:
                    self.history_best_fit = self.fits[i]
                    self.history_best_x = self.xs[i].copy()
                    self.best_update()

    def update_paricle(self, i, d):
        delta_v = self.c1[self.step_num] * np.random.uniform(0, 1) * (
                self.atom_history_best_x[self.fid[i, d], d] - self.xs[i, d])
        new_v = self.w1[self.step_num] * self.vs[i, d] + delta_v
        if new_v > self.max_v:
            new_v = self.max_v
        if new_v < self.min_v:
            new_v = self.min_v
        self.vs[i, d] = new_v
        x = self.xs[i, d]
        x += new_v
        self.xs[i, d] = x


if __name__ == '__main__':
    s = EpsoSwarm(500, 20, True, fun, 10, 100, -100, None)
    s.run()
    # print('best fit', test_fun(np.zeros(10)))
