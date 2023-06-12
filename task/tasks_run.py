import copy
import json
import traceback

import numpy as np
import pickle

from env.NormalEnv import NormalEnv
from rl.DDPG.TF2_DDPG_Basic import DDPG
from settings import *
from task.task_run_utils.common import get_task_result, result_process, get_tasks_result
from task.task_run_utils.top_task_run import top_task_run
from task.utils.all_task_final_result_process.all_task_final_result_process import all_task_final_result_process
from task.utils.evluate_optimizer import evluate_optimizer
from train.ddpg import get_ddpg_object
from utils.task_hash import get_task_hash, task2str, get_task_hashs
from log import logger
from task.task_run_utils.result_evaluate_task import result_evaluate_task_run, new_result_evaluate_task_run
import time


def task_run(task, mq=None):  # 用于多进程运行
    task_md5 = get_task_hash(task)
    logger.info(f"运行任务{task_md5}-{task.get('type')}-{task}")

    result = get_task_result(task) if task['type'] not in ['top', 'new_result_evaluate'] else None
    # result = get_task_result(task)
    try:
        if result:
            logger.info(F'{task_md5} cache find,return')
            return result_process(task, result, write=False, mq=mq)
        if task['type'] == 'all':
            return all_task_run(task, mq)
        elif task['type'] == 'train':
            return train_task_run(task, mq)
        elif task['type'] == 'single_train':
            return single_train_task_run(task, mq)
        elif task['type'] == 'evaluate_models':
            return evaluate_models_task_run(task, mq)
        elif task['type'] == 'evaluate_multi_times':
            return evaluate_multi_times_task_run(task, mq)
        elif task['type'] == 'single_evaluate':
            return single_evaluate_task_run(task, mq)
        elif task['type'] == 'result_evaluate':
            return result_evaluate_task_run(task, mq)
        elif task['type'] == 'new_result_evaluate':
            return new_result_evaluate_task_run(task, mq)
        elif task['type'] == 'top':
            return top_task_run(task, mq)
        else:
            raise ValueError(f"{task['type']} 未定义")
    except Exception as e:
        with open('error.txt', 'a') as f:
            traceback.print_exc(file=f)
        logger.info(f"error-print-{task_md5}-{task.get('type')}-{task}")
        traceback.print_exc()
        logger.info(f"error-print-end{task_md5}-{task.get('type')}-{task}")
        time.sleep(20)
        raise e


def all_task_run(task, mq=None):
    assert task['type'] == 'all', '只接受类型为all的任务'

    task_md5 = get_task_hash(task)

    optimizer = task['evaluate_optimizer']
    train_max_episode = task['train_max_episode']
    train_max_steps = task['train_max_steps']
    fun_nums = task['evaluate_function']
    group = task['group']
    train_num = task['train_times']
    runtimes = task['runtimes']
    separate_train = task['separate_train']
    max_fe = task['max_fe']
    dim = task['dim']
    base_evaluate_optimizers = task['base_evaluate_optimizers']
    lr_critic = task.get('lr_critic', 1e-7)
    lr_actor = task.get('lr_actor', 1e-9)
    # 生成训练子任务
    tasks = []

    train_task_dic = {
        'type': 'train',
        'optimizer': optimizer,
        'group': group,
        'train_max_steps': train_max_steps,
        'train_max_episode': train_max_episode,
        'fun_nums': fun_nums,
        'train_num': train_num,
        'separate_train': separate_train,
        'runtimes': runtimes,
        'dim': dim,
        'max_fe': max_fe,
        'n_part': task['n_part'],
        'lr_critic': lr_critic,
        'lr_actor': lr_actor,
    }

    tasks = [train_task_dic]

    # 训练任务结果收集
    results = get_tasks_result(tasks)

    # 如无结果则等待结果
    if results is None:
        logger.info(f"任务 {task_md5} 需要补充信息")
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': tasks
        }
        return result_process(task, task_result, write=False, mq=mq)

    # 读取结果  需要  每个算法在一起/分别训练 分群数不同 等情况下的训练好的算法
    train_result = results[0]['result']  # {f_num:[models]}

    # 生成评估子任务
    tasks = []
    new_task = {
        'type': 'result_evaluate',
        'optimizer': optimizer,
        'group': group,
        'base_evaluate_optimizer': base_evaluate_optimizers,
        'separate_train': separate_train,
        'runtimes': runtimes,
        'dim': dim,
        'max_fe': max_fe,
        'model': train_result,
        'n_part': task['n_part'],
    }
    tasks.append(new_task)

    # 评估任务结果收集
    results = get_tasks_result(tasks)

    # 如无结果则等待结果
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': tasks
        }
        return result_process(task, task_result, write=False, mq=mq)

    # 读取结果  需要  每个算法在一起/分别训练 分群数不同 等情况下不同的平均排名和曲线

    rank_result = None
    display_result = None
    '''
    
    '''
    task_result = copy.deepcopy(task)
    task_result['result'] = []
    for result in results:
        task_result['result'].append(all_task_final_result_process(result, task['evaluate_optimizer']))
    task_result['md5'] = get_task_hash(task)
    task_result['train_result'] = train_result

    return result_process(task, task_result, mq)


train_task_test_dic = {
    'optimizer': None,
    'group': 5,
    'train_max_steps': 0,
    'train_max_episode': 0,
    'fun_nums': [1, ],
    'train_num': 3,
    'separate_train': True,
    'runtimes': 10,
    'dim': 20,
    'max_fe': 1e4,
}


def train_task_run(task, mq=None):
    assert task['type'] == 'train', '只接受类型为train的任务'
    '''
    需要 优化算法 运行次数 是否单独训练 维度 分群 训练次数
    
    最终结果为 一个算法 三个模型 如果分开 则是一个算法 3N个模型
    '''
    optimizer = task['optimizer']
    train_max_episode = task['train_max_episode']
    train_max_steps = task['train_max_steps']
    fun_nums = task['fun_nums']
    group = task['group']
    train_num = task['train_num']
    runtimes = task['runtimes']
    max_fe = task['max_fe']
    dim = task['dim']
    lr_critic = task.get('lr_critic', 1e-7)
    lr_actor = task.get('lr_actor', 1e-9)

    tasks = []
    if task['separate_train']:
        for fun_num in fun_nums:
            single_train_task_dic = {
                'type': 'single_train',
                'optimizer': optimizer,
                'group': group,
                'train_max_steps': train_max_steps,
                'train_max_episode': train_max_episode,
                'fun_nums': [fun_num, ],
                'train_num': train_num,
                'runtimes': runtimes,
                'dim': dim,
                'max_fe': max_fe,
                'n_part': task['n_part'],
                'lr_critic': lr_critic,
                'lr_actor': lr_actor,
            }
            tasks.append(single_train_task_dic)
    else:
        single_train_task_dic = {
            'type': 'single_train',
            'optimizer': optimizer,
            'group': group,
            'train_max_steps': train_max_steps,
            'train_max_episode': train_max_episode,
            'fun_nums': fun_nums,
            'train_num': train_num,
            'runtimes': runtimes,
            'dim': dim,
            'max_fe': max_fe,
            'n_part': task['n_part'],
            'lr_critic': lr_critic,
            'lr_actor': lr_actor,
        }
        tasks.append(single_train_task_dic)

    results = get_tasks_result(tasks)

    # 如无结果则等待结果
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': tasks
        }
        return result_process(task, task_result, write=False, mq=mq)

    real_result = {}
    for result in results:
        models = result['result']
        res_f_nums = result['fun_nums']
        for res_f_num in res_f_nums:
            real_result[res_f_num] = models

    task_result = copy.deepcopy(task)
    task_result['result'] = real_result
    task_result['md5'] = get_task_hash(task)
    return result_process(task, task_result, mq)


single_train_task_test_dic = {
    'optimizer': None,
    'group': 5,
    'train_max_steps': 0,
    'train_max_episode': 0,
    'fun_nums': [1, ],
    'train_num': 3,
    'runtimes': 10,
    'dim': 20,
    'max_fe': 1e4,
}


def single_train_task_run(task, mq=None):
    assert task['type'] == 'single_train', '只接受类型为train的任务'
    optimizer = task['optimizer']
    train_max_episode = task['train_max_episode']
    train_max_steps = task['train_max_steps']
    fun_nums = task['fun_nums']
    group = task['group']
    runtimes = task['runtimes']
    max_fe = task['max_fe']
    dim = task['dim']
    n_part = task['n_part']
    lr_critic = task.get('lr_critic', 1e-7)
    lr_actor = task.get('lr_actor', 1e-9)
    # 训练过程
    gym_env = NormalEnv(obs_shape=(optimizer.obs_space,), action_shape=(optimizer.action_space * group,),
                        target_optimizer=optimizer, fun_nums=fun_nums, max_fe=max_fe, n_part=n_part)

    assert (gym_env.action_space.high == -gym_env.action_space.low)
    is_discrete = False
    task_md5 = get_task_hash(task)
    task_dir = TASK_PATH.joinpath(f'{task_md5}/')
    for i in range(task['train_num']):
        if os.path.exists(task_dir.joinpath(f"ddpg_actor_final_round{i}.h5")):
            continue
        ddpg = get_ddpg_object(gym_env, discrete=is_discrete, memory_cap=10000000, lr_critic=lr_critic,
                               lr_actor=lr_actor)
        save_freq = train_max_episode / 20
        if save_freq < 1:
            save_freq = 1
        save_freq = int(save_freq)
        ddpg.train(max_episodes=train_max_episode, max_epochs=train_max_steps, task_path=task_dir, train_num=i,
                   save_freq=save_freq)

    # 评估过程
    all_models = task_dir.glob('ddpg_actor*.h5')
    new_task = {
        'type': 'evaluate_models',
        'evaluate_optimizers': [
        ],
        'evaluate_functions': fun_nums,
        'dims': [dim, ],
        'groups': [group, ],
        'runtimes': runtimes if runtimes < 5 else 5,
        'max_fe': max_fe,
        'n_part': task['n_part'],
    }
    for model in all_models:
        evaluate_optimizer = {
            'optimizer': optimizer,
            'model': model
        }
        new_task['evaluate_optimizers'].append(evaluate_optimizer)

    tasks = [new_task]

    results = get_tasks_result(tasks)

    # 如无结果则等待结果
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': tasks
        }
        return result_process(task, task_result, write=False, mq=mq)

    real_results = results[0]['result'][:3]
    new_models = [real_result['model'] for real_result in real_results]

    task_result2 = copy.deepcopy(task)
    task_result2['result'] = new_models
    task_result2['md5'] = get_task_hash(task)

    return result_process(task, task_result2, mq)


evaluate_task_test_dic = {
    'evaluate_optimizers': [
        {
            'optimizer': None,
            'model': 'model_path'
        },
    ],
    'evaluate_functions': [1, ],
    'dims': [20, ],
    'groups': [5, ],
    'run_times': 1,
    'max_fe': 1e4,
}


def evaluate_models_task_run(task, mq=None):
    assert task['type'] == 'evaluate_models', '只接受类型为evaluate的任务'
    '''
    需要 优化算法(模型) 运行次数 维度 分群

    最终结果为 算法排名  算法运行结果
    '''

    # 生成任务
    tasks = []
    for evaluate_optimizer in task['evaluate_optimizers']:
        for evaluate_function in task['evaluate_functions']:
            for dim in task['dims']:
                for group in task['groups']:
                    single_evaluate_task = {
                        'type': 'evaluate_multi_times',
                        'evaluate_optimizer': evaluate_optimizer['optimizer'],
                        'model': evaluate_optimizer['model'],
                        'evaluate_function': evaluate_function,
                        'dim': dim,
                        'group': group,
                        'max_fe': task['max_fe'],
                        'runtimes': task['runtimes'],
                        'n_part': task['n_part'],
                    }
                    tasks.append(single_evaluate_task)

    # 结果检测
    results = get_tasks_result(tasks)

    # 如无结果则等待结果
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': tasks
        }
        return result_process(task, task_result, write=False, mq=mq)

    # 如有结果则组织最终结果
    results.sort(key=lambda result: result['result'][-1][2])

    task_result = copy.deepcopy(task)
    task_result['result'] = results
    task_result['md5'] = get_task_hash(task)

    return result_process(task, task_result, mq)


def evaluate_multi_times_task_run(task, mq=None):
    assert task['type'] == 'evaluate_multi_times', '只接受类型为evaluate的任务'
    # 生成任务
    tasks = []
    for i in range(task['runtimes']):
        copy_task = copy.deepcopy(task)
        copy_task['type'] = 'single_evaluate'
        del copy_task['runtimes']
        copy_task['run_index'] = i
        tasks.append(copy_task)

    # 结果检测
    results = get_tasks_result(tasks)

    # 如无结果则等待结果
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': tasks
        }
        return result_process(task, task_result, write=False, mq=mq)

    ress = []
    for result in results:
        ress.append(result['result'])
    ress = np.array(ress)
    average_ress = np.average(ress, axis=0)

    task_result = copy.deepcopy(task)
    task_result['result'] = average_ress
    task_result['md5'] = get_task_hash(task)

    return result_process(task, task_result, mq)


single_evaluate_task_test_dic = {
    'evaluate_optimizer': None,
    'model': 'model_path',
    'evaluate_function': 1,
    'dim': 20,
    'group': 5,
    'run_index': 1,
    'max_fe': 1e4,
}


def single_evaluate_task_run(task, mq=None):
    assert task['type'] == 'single_evaluate', '只接受类型为single_evaluate的任务'
    '''
    需要 优化算法(模型) 维度 分群 评估函数

    最终结果为  算法运行结果
    '''

    task_result = get_task_result(task)
    if not task_result:
        # 运行获取结果
        result = evluate_optimizer(task)
        task_result = copy.deepcopy(task)
        task_result['result'] = result
        task_result['md5'] = get_task_hash(task)

    return result_process(task, task_result, mq)
