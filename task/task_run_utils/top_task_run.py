import copy
from pathlib import Path

from settings import TASK_PATH
from task.task_run_utils.common import get_tasks_result, result_process
from utils.db.db import save_optimizer
from utils.task_hash import get_task_hash, get_task_hashs
from log import logger
import pandas as pd


def top_task_run(task, mq=None):
    tasks = []

    evaluate_optimizers = task['evaluate_optimizers']
    base_evaluate_optimizers = task['base_evaluate_optimizers']
    runtimes = task['runtimes']
    separate_trains = task['separate_trains']
    groups = task['groups']
    train_max_episode = task['train_max_episode']
    train_max_steps = task['train_max_steps']
    dims = task['dims']
    evaluate_function = task['evaluate_function']
    lr_critic = task.get('lr_critic', 1e-7)
    lr_actor = task.get('lr_actor', 1e-9)

    for evaluate_optimizer in evaluate_optimizers:
        for separate_train in separate_trains:
            for group in groups:
                for dim in dims:
                    new_task = {
                        'type': 'all',
                        'base_evaluate_optimizers': base_evaluate_optimizers,
                        'runtimes': runtimes,
                        'separate_train': separate_train,
                        'evaluate_function': evaluate_function,
                        'train_max_episode': train_max_episode,
                        'train_max_steps': train_max_steps,
                        'max_fe': 1e4,
                        'n_part': 100,
                        'dim': dim,
                        'group': group,
                        'evaluate_optimizer': evaluate_optimizer,
                        'train_times': task['train_times'],
                        'lr_critic': lr_critic,
                        'lr_actor': lr_actor,
                    }
                    tasks.append(new_task)

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

    task_results = []
    for result in results:
        task_result = {
            'type': f"{result['evaluate_optimizer'].optimizer_name}-separate_train{result['separate_train']}-group{result['group']}-"
                    f"dim{result['dim']}",
            'result': result['result'],
            'train_result': result['train_result'],
        }
        task_results.append(task_result)
        # runtimes = result['runtimes']
        # separate_train = result['separate_train']
        # train_max_episode = result['train_max_episode']
        # train_max_steps = result['train_max_steps']
        # max_fe = result['max_fe']
        # n_part = result['n_part']
        # dim = result['dim']
        # group = result['group']
        # train_times = result['train_times']
        # lr_critic = result['lr_critic']
        # lr_actor = result['lr_actor']

    xlsx_result = []
    for result in results:
        d = copy.deepcopy(result)
        del d['base_evaluate_optimizers']
        del d['evaluate_function']
        d1 = d.pop('result')
        d['train_average_rank'] = d1[0]['train_average_rank']
        d['origin_average_rank'] = d1[0]['origin_average_rank']
        d['optimizer'] = d.pop('evaluate_optimizer').optimizer_name
        d['train_result'] = d.pop('train_result')
        xlsx_result.append(d)
    save_optimizer(xlsx_result)
    df = pd.DataFrame(xlsx_result)
    # print(df)
    writer = pd.ExcelWriter('最终结果.xlsx')
    df.to_excel(writer, 'Sheet1')  # 这里假设df是一个pandas的dataframe
    writer.save()
    writer.close()

    task_result = copy.deepcopy(task)
    task_result['result'] = task_results
    task_result['md5'] = get_task_hash(task)

    logger.info(f"最终结果路径 :{task_result['md5']}")

    return result_process(task, task_result, mq)
