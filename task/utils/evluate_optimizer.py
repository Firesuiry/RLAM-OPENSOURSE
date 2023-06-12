import json
import os
import numpy as np

from functions import CEC_functions
from matAgent.hrlepso import HrlepsoSwarm
from evaluate.common import *


def evluate_optimizer(task):
    print(f'evluate_optimizer config:{task}')
    dim = task['dim']
    cls = task['evaluate_optimizer']
    model = task.get('model')
    npart = task['n_part']
    f_num = task['evaluate_function']
    max_fe = task.get('max_fe')
    group = task['group']

    cec_functions = CEC_functions(dim)

    def test_fun(x):
        if len(x.shape) == 2:
            ans = []
            for row in x:
                y = cec_functions.Y(row, f_num)
                ans.append(y)
        else:
            ans = cec_functions.Y(x, f_num)
        return np.array(ans)

    config_dic = {
        'model': model,
        'group': group,
    }
    if max_fe:
        config_dic['max_fes'] = max_fe

    optimizer = cls(n_run=260, n_part=npart, show=False, fun=test_fun, n_dim=dim, pos_max=100, pos_min=-100,
                    config_dic=config_dic)
    optimizer.run()

    res = optimizer.result_cache

    if len(res) == 0:
        print(locals())
        raise BaseException('没有获取到结果')
    # print(
    #     f'{optimizer.name}|gbest:{optimizer.get_best_value()}|best:{best_value}|fes:{optimizer.max_fes}|npart:{optimizer.npart}|ndim:{optimizer.ndim}')
    return res


if __name__ == '__main__':
    config = {'dim': 10, 'class': HrlepsoSwarm,
              'model': 'D:\\develop\\swam\\ieeeaccess - 副本 - 副本 - 副本 - 副本\\rl\\train0\\ddpg_actor_episode100.h5',
              'npart': 20, 'f_num': 1, 'task': 280, 'task_md5': 'fbd5109e056c85dd0c82dc0172ed1aab'}
    evluate_optimizer(config)
