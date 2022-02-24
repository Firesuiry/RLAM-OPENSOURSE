import json
import os
import pickle

from settings import TASK_PATH
from utils.task_hash import get_task_hash, task2str, get_task_hashs
from log import logger


def result_process(task, result, mq=None, result_num='', write=True):
    task_md5 = get_task_hash(task)
    # logger.info(f'任务{task_md5}-结束 结果:{result}')

    task_dir = TASK_PATH.joinpath(f'{task_md5}/')
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    if write:
        # json保存
        # result_path = task_dir.joinpath(f'result{result_num}.json')
        # json_str = task2str(result)
        # with open(result_path, 'w', encoding='UTF-8') as f:
        #     f.write(json_str)

        task_path = task_dir.joinpath(f'task.json')
        task_json_str = task2str(task)
        with open(task_path, 'w', encoding='UTF-8') as f:
            f.write(task_json_str)
            if task['type'] == 'top':
                b = 1
                pass

        # pickle保存
        # save_path_pickle = task_dir.joinpath(f'task.pickle')
        # with open(save_path_pickle, 'wb') as f:
        #     pickle.dump(task, f, 0)

        result_path_pickle = task_dir.joinpath(f'result{result_num}.pickle')
        with open(result_path_pickle, 'wb') as f:
            pickle.dump(result, f, 0)
        logger.info(f'result write into:{result_path_pickle}')
    if mq:
        logger.info(f'queue size:{mq.qsize()}')
        mq.put(result)
        return

    return result


def get_task_result(task, result_num=''):
    task_md5 = get_task_hash(task)
    task_dir = TASK_PATH.joinpath(f'{task_md5}/')
    result_path = task_dir.joinpath(f'result{result_num}.pickle')
    # logger.info(result_path)
    if os.path.exists(result_path):
        # logger.info(f'{task_md5} cache yes')
        with open(result_path, 'rb') as f:
            return pickle.load(f)
    else:
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
            # task_path = task_dir.joinpath(f'task.json')
            # task_json_str = task2str(task)
            # with open(task_path, 'w', encoding='UTF-8') as f:
            #     f.write(task_json_str)
        # logger.info(f'{task_md5} cache no')
        return None


def get_tasks_result(tasks):
    results = []
    for task in tasks:
        result = get_task_result(task)
        if result is None:
            results = None
            break
        results.append(result)
    # logger.debug(f'获取任务 {get_task_hashs(tasks)} 结果:{results}')
    return results
