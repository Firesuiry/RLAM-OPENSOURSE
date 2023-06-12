from matAgent.baseAgent import *

# CLPSO

# Comprehensive learning PSO for global optimization of multimodal functions

# 线性变化权重


W0 = 0.9
W1 = 0.4
# 多次没优化阈值
M = 7
# 速度变化系数
C = 1.49445


class ClpsoSwarm(MatSwarm):
    optimizer_name = 'CLPSO'
    action_space = 4

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'CLPSO'
        self.fid = np.zeros((self.n_part, self.n_dim), dtype=np.int)
        self.vs = np.zeros_like(self.xs)
        self.atom_history_best_x = np.zeros((self.n_part, self.n_dim))
        self.atom_history_best_fit = np.zeros(self.n_part)
        self.atom_pci = np.zeros(self.n_part)
        self.flag = np.zeros(self.n_part)

        self.max_v = 0.2 * (pos_max - pos_min)
        self.min_v = -self.max_v

        self.init()

    def init(self):
        print('初始化')
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.min_v, self.max_v, self.xs.shape)

        indexs = np.array(list(range(self.n_part)))
        self.atom_pci = 0.05 + 0.45 * np.exp(10 * indexs / (self.n_part - 1)) / (np.exp(10) - 1)
        self.fits = self.fun(self.xs)
        self.atom_history_best_x = self.xs.copy()
        self.atom_history_best_fit = self.fits.copy()

        best_index = np.argmin(self.fits)
        self.history_best_x = self.xs[best_index].copy()
        self.history_best_fit = self.fits[best_index].copy()

        for i in range(self.n_part):
            self.caculate_fid(i)
        self.init_finish = True

    def caculate_fid(self, i):
        #  Selection of exemplar dimensions for particle i.
        rands = np.random.uniform(0, 1, self.n_dim)
        for d in range(self.n_dim):
            rand = rands[d]
            if rand < self.atom_pci[i]:
                fids = np.random.randint(0, self.n_part, 2)
                fits = self.atom_history_best_fit[fids]
                self.fid[i, d] = fids[np.argmin(fits)]
            else:
                self.fid[i, d] = i

    def run_once(self, actions=None, *args, **kwargs):

        # 迭代
        w = W0 + (W1 - W0) * (self.step_num + 1) / self.n_run
        c1 = C
        # print(w)
        # print('iter:{} best:{}'.format(self.step_num, self.history_best_fit))
        for i in range(self.n_part):
            if actions is not None:
                w, other_coefficient, mutation_rate = self.get_coefficients(actions, i)
                c1 = other_coefficient[0]
            # 粒子
            if self.flag[i] >= M:
                self.caculate_fid(i)
                self.flag[i] = 0
            for d in range(self.n_dim):
                # 维度
                target = self.atom_history_best_x[self.fid[i, d], d]
                new_v = w * self.vs[i, d] + c1 * np.random.uniform(0, 1) * (target - self.xs[i, d])
                if new_v > self.max_v:
                    new_v = self.max_v
                if new_v < self.min_v:
                    new_v = self.min_v
                self.vs[i, d] = new_v
                x = self.xs[i, d]
                x += new_v
                self.xs[i, d] = x
            self.fits[i] = self.fun(self.xs[i])
            x = self.xs[i]
            if (x > self.pos_min).all() and (x < self.pos_max).all():
                # 表示在范围内
                if self.fits[i] < self.atom_history_best_fit[i]:
                    # 找到新的最优值
                    # print('找到新的最优值')
                    self.atom_history_best_fit[i] = self.fits[i]
                    self.atom_history_best_x[i] = self.xs[i]
                    self.flag[i] = 0
                    if self.fits[i] < self.history_best_fit:
                        self.history_best_fit = self.fits[i]
                        self.history_best_x = self.xs[i].copy()
                        self.best_update()
                else:
                    self.flag[i] += 1

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
        gbestx = self.history_best_x[0]
        gbesty = self.history_best_x[1]
        # print(self.history_best_x)
        plt.scatter(gbestx, gbesty, alpha=1, edgecolors='black')
        plt.xlim(self.pos_min, self.pos_max)
        plt.ylim(self.pos_min, self.pos_max)
        plt.pause(0.05)


if __name__ == '__main__':
    show = True
    # s = ClpsoSwarm(100, 10, show, fun, 2, pos_max=10, pos_min=-10, config_dic=None)
    s = ClpsoSwarm(100, 20, show, fun, 2, 100, -100, None)
    s.run()
