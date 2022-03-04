import copy
from pathlib import Path

from task.result_evaluate_task import visible_res_generate
from task.task_run_utils.common import get_tasks_result, result_process
from utils.task_hash import get_task_hash, get_task_hashs

result_evaluate_task_test_dic = {
    'type': 'result_evaluate',
    'optimizer': None,
    'model': {
        1: [
            'model_path1',
        ]
    },
    'base_evaluate_optimizer': [None, ],
    'evaluate_function': list(range(1, 29, 1)),
    'group': 1,
    'max_fe': 1e4,
    'n_part': 100,
    'dim': 20,
    'runtimes': 10,
}


def result_evaluate_task_run(task, mq=None):
    assert task['type'] == 'result_evaluate'
    visiable_result_flag = task.get('show')

    tasks = []

    for optimizer in task['base_evaluate_optimizer']:
        for f_num, models in task['model'].items():
            single_evaluate_task = {
                'type': 'evaluate_multi_times',
                'evaluate_optimizer': optimizer,
                'model': None,
                'evaluate_function': f_num,
                'dim': task['dim'],
                'group': task['group'],
                'max_fe': task['max_fe'],
                'runtimes': task['runtimes'],
                'n_part': task['n_part'],
            }
            tasks.append(single_evaluate_task)

    for iter_f_num, models in task['model'].items():
        for model in models:
            single_evaluate_task = {
                'type': 'evaluate_multi_times',
                'evaluate_optimizer': task['optimizer'],
                'model': model,
                'evaluate_function': iter_f_num,
                'dim': task['dim'],
                'group': task['group'],
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

    # 对多个模型的 进行择优处理
    best_cache = {}
    new_results = []
    new_result2 = {}
    for result in results:
        if result['evaluate_optimizer'].optimizer_name == task['optimizer'].optimizer_name and result.get('model'):
            f_num = result['evaluate_function']
            new_result2[f_num] = {}
            if best_cache.get(f_num):
                cache_result = best_cache[f_num]
                if cache_result['result'][-1][2] > result['result'][-1][2]:
                    best_cache[f_num] = result
            else:
                best_cache[f_num] = result
        else:
            new_results.append(result)
    for result in best_cache.values():
        new_results.append(result)

    if visiable_result_flag:
        # 可视化结果
        pass

    for result in new_results:
        info = '-'
        if result['model']:
            info += 'train'
        else:
            info += 'origin'

        key = result['evaluate_optimizer'].optimizer_name + info
        new_result2[result['evaluate_function']][key] = result

    task_result = copy.deepcopy(task)
    task_result['result'] = new_result2
    task_result['md5'] = get_task_hash(task)

    return result_process(task, task_result, mq)


new_result_evaluate_task_test_dic = {
    'type': 'new_result_evaluate',
    'optimizer_model_list': [
        {
            'optimizer': 'optimizer_class',
            'fun_model': {
                1: [
                    'model_path1',
                ]
            }
        }
    ],
    'evaluate_function': list(range(1, 29, 1)),
    'group': 1,
    'max_fe': 1e4,
    'n_part': 100,
    'dim': 20,
    'runtimes': 10,
}


def new_result_evaluate_task_run(task, mq=None):
    assert task['type'] == 'new_result_evaluate'
    visiable_result_flag = task.get('show')

    tasks = []

    for optimizer_model in task['optimizer_model_list']:
        optimizer = optimizer_model['optimizer']
        fun_model = optimizer_model.get('fun_model')
        for fun_num, models in fun_model.items():
            for model in models:
                single_evaluate_task = {
                    'type': 'evaluate_multi_times',
                    'evaluate_optimizer': optimizer,
                    'model': model,
                    'evaluate_function': fun_num,
                    'dim': task['dim'],
                    'group': task['group'],
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

    fun_optimizer_result_dict = {}
    for result in results:
        f_num = result['evaluate_function']
        optimizer_name = result['evaluate_optimizer'].optimizer_name
        if result.get('model'):
            optimizer_name += 'train'
        else:
            optimizer_name += 'origin'
        if f_num not in fun_optimizer_result_dict:
            fun_optimizer_result_dict[f_num] = {}
        if optimizer_name not in fun_optimizer_result_dict[f_num]:
            fun_optimizer_result_dict[f_num][optimizer_name] = result
        else:
            if fun_optimizer_result_dict[f_num][optimizer_name]['result'][-1][2] > result['result'][-1][2]:
                fun_optimizer_result_dict[f_num][optimizer_name] = result

    if visiable_result_flag or True:
        # 可视化结果
        visible_res_generate(fun_optimizer_result_dict)

    return_result = {}
    for f_num in fun_optimizer_result_dict.keys():
        if f_num not in return_result:
            return_result[f_num] = {}
        for optimizer_name in fun_optimizer_result_dict[f_num].keys():
            return_result[f_num][optimizer_name] = fun_optimizer_result_dict[f_num][optimizer_name]

    task_result = copy.deepcopy(task)
    task_result['result'] = return_result
    task_result['md5'] = get_task_hash(task)

    return result_process(task, task_result, mq)
