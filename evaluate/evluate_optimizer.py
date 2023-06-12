import json
import os
import numpy as np

from functions import CEC_functions
from matAgent.hrlepso import HrlepsoSwarm
from evaluate.common import *


def evluate_optimizer(config):
    print(f'evluate_optimizer config:{config}')
    task_ID = config.get('task')
    task_md5 = config.get('task_md5')
    dim = config['dim']
    cls = config['class']
    model = config.get('model')
    npart = config['npart']
    f_num = config['f_num']
    max_fe = config.get('max_fe')
    file_name = TASK_RES_DIR.joinpath('{}/task{}.json'.format(task_md5, task_ID))
    if os.path.exists(file_name):
        try:
            with open(file_name, 'r', encoding='UTF-8') as f:
                json_str = f.read()
                res = json.loads(json_str)
                return res
        except Exception as e:
            print('get cache error:', e, file_name)

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
    }
    if max_fe:
        config_dic['max_fes'] = max_fe

    optimizer = cls(n_run=260, n_part=npart, show=False, fun=test_fun, n_dim=dim, pos_max=100, pos_min=-100,
                    config_dic=config_dic)
    optimizer.run()

    res = {
        'record': optimizer.result_cache,
        # 'gbest': optimizer.get_best_value(),
        'task': task_ID,
        'class': optimizer.name,
        'dim': dim,
        'f_num': f_num
    }
    if len(res['record']) == 0:
        print(locals())
        raise BaseException('没有获取到结果')
    # print(
    #     f'{optimizer.name}|gbest:{optimizer.get_best_value()}|best:{best_value}|fes:{optimizer.max_fes}|npart:{optimizer.npart}|ndim:{optimizer.ndim}')

    json_str = json.dumps(res)
    with open(file_name, 'w', encoding='UTF-8') as f:
        f.write(json_str)
    return res


if __name__ == '__main__':
    config = {'dim': 10, 'class': HrlepsoSwarm,
              'model': 'D:\\develop\\swam\\ieeeaccess - 副本 - 副本 - 副本 - 副本\\rl\\train0\\ddpg_actor_episode100.h5',
              'npart': 20, 'f_num': 1, 'task': 280, 'task_md5': 'fbd5109e056c85dd0c82dc0172ed1aab'}
    evluate_optimizer(config)
