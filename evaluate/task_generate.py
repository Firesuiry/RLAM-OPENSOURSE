import copy
import glob
import hashlib
import json
import multiprocessing as mp
import os

from data_process.data_process import data_process
from evaluate.common import *
from evaluate.evluate_optimizer import evluate_optimizer
from matAgent.clpso import ClpsoSwarm
from matAgent.epso import EpsoSwarm
from matAgent.fdrpso import FdrpsoSwarm
from matAgent.hpso_tvac import HpsotvacSwarm
from matAgent.lips import LipsSwarm
from matAgent.olpso import OlpsoSwarm
from matAgent.pso import PsoSwarm
from matAgent.rlepso import RlepsoSwarm
from matAgent.shpso import ShpsoSwarm

pso_swarms = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, OlpsoSwarm, PsoSwarm, ShpsoSwarm, EpsoSwarm]


def md5(s):
    m = hashlib.md5()
    b = s.encode(encoding='utf-8')
    m.update(b)
    str_md5 = m.hexdigest()
    return str_md5


def generate_all_task(dims, models=None, runtimes=1, rl_swarm_model=None):
    tasks = []
    tasks += generate_norl_task(dims, runtimes, pso_swarms)
    if models:
        tasks += generate_rl_task(dims, models, runtimes, RlepsoSwarm)
    if rl_swarm_model:
        for swarm, rlmodels in rl_swarm_model.items():
            tasks += generate_rl_task(dims, rlmodels, runtimes, swarm)
    return tasks


def generate_rl_task(dims, models, runtimes, rl_optimizer, test_funcs=None):
    if test_funcs is None:
        test_funcs = list(range(1, 29, 1))
    tasks = []
    if models is None:
        return []
    for test_func in test_funcs:
        for dim in dims:
            for model in models:
                task = {
                    'dim': dim,
                    'class': rl_optimizer,
                    'model': model,
                    'npart': 100,
                    'f_num': test_func
                }
                for _ in range(runtimes):
                    tasks.append(task.copy())
    return tasks


def generate_norl_task(dims, runtimes, optimizers, test_funcs=None):
    if test_funcs is None:
        test_funcs = list(range(1, 29, 1))
    tasks = []
    for test_func in test_funcs:
        for dim in dims:
            for optimizer in optimizers:
                task = {
                    'dim': dim,
                    'class': optimizer,
                    'npart': 100,
                    'f_num': test_func
                }
                for _ in range(runtimes):
                    tasks.append(task.copy())
    return tasks


def evaluate_model(dims, processes=4, models=None, runtimes=1, rl_swarm_models=None):
    tasks = generate_all_task(dims, models, runtimes, rl_swarm_model=rl_swarm_models)

    for i in range(len(tasks)):
        tasks[i]['task'] = i

    json_tasks = copy.deepcopy(tasks)
    for i in range(len(json_tasks)):
        json_tasks[i]['class'] = json_tasks[i]['class'].__name__
    task_str = json.dumps(json_tasks)
    task_md5 = md5(task_str)

    # 设置task目录
    if not os.path.exists(TASK_RES_DIR):
        os.mkdir(TASK_RES_DIR)
    dir_path = '{}/{}/'.format(TASK_RES_DIR, task_md5)
    if os.path.exists(dir_path):
        pass
    else:
        os.mkdir(dir_path)
    with open('{}/task.json'.format(dir_path), 'w') as f:
        f.write(task_str)

    for i in range(len(tasks)):
        tasks[i]['task_md5'] = task_md5

    if processes and processes > 1:
        print('多进程')
        pool = mp.Pool(processes=processes)
        ress = pool.map(evluate_optimizer, tasks)
    else:
        print('单进程')
        ress = []
        for task in tasks:
            ress.append(evluate_optimizer(task))

    data = {}

    for res in ress:
        dim = res['dim']
        name = res['class']
        f_num = res['f_num']
        set_dict(data, [dim, f_num, name], res)

    data_process(data)
    json_str = json.dumps(data)
    with open(f'{dir_path}/json.json', 'w') as f:
        f.write(json_str)


def set_dict(dic, keys, value):
    for key in keys[:-1]:
        if key in dic:
            dic = dic[key]
        else:
            dic[key] = {}
            dic = dic[key]
    key = keys[-1]
    if key in dic:
        pass
    else:
        dic[key] = []
    dic[key].append(value)


if __name__ == '__main__':
    evaluate_model(dims=[20], processes=8, models=None, runtimes=1)
