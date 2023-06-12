import os.path

import numpy as np
from pathlib import Path
import json
import pickle


def get_task_by_type(task_type):
    for task_file in Path('task').glob('*'):
        result_file = task_file.joinpath('result.pickle')
        task_json_file = task_file.joinpath('task.json')
        if not os.path.exists(result_file) or not os.path.exists(task_json_file):
            continue
        with open(task_json_file, 'r') as f:
            task_des = json.loads(f.read())
        print(task_des.get('type'))
        if task_des['type'] == task_type:
            with open(result_file, 'rb') as f:
                task_result = pickle.load(f)
            return task_des, task_result


def delete_not_single_evaluate_task():
    for task_file in Path('task').glob('*'):
        result_file = task_file.joinpath('result.pickle')
        task_json_file = task_file.joinpath('task.json')
        if not os.path.exists(result_file) or not os.path.exists(task_json_file):
            continue
        with open(task_json_file, 'r') as f:
            task_des = json.loads(f.read())
        if task_des['type'] not in ['single_evaluate', 'single_train'] or True:
            print('delete', task_file)
            for d_file in task_file.glob('*'):
                os.remove(d_file)
            task_file.rmdir()
        # if task_des['type'] in ['single_evaluate', 'single_train']:
        #     # print(task_des)
        #     if (task_des.get('evaluate_optimizer') or task_des.get('optimizer')) in ['SHPSO', 'GWO']:
        #         print('delete single_evaluate', task_file)
        #         for d_file in task_file.glob('*'):
        #             os.remove(d_file)
        #         task_file.rmdir()


def delete_all_task():
    i = 0
    for task_file in Path('task').glob('*'):
        print(F'{i} delete', task_file)
        i += 1
        for d_file in task_file.glob('*'):
            os.remove(d_file)
        task_file.rmdir()
    os.remove('../db.db')


swarms = []


def delete_swarm_task():
    i = 0
    for task_file in Path('task').glob('*'):
        task_json_file = task_file.joinpath('task.json')
        if os.path.exists(task_json_file):
            with open(task_json_file, 'r') as f:
                task_des = json.loads(f.read())
            swarm_name = task_des.get('evaluate_optimizer', '')
            if swarm_name not in swarms:
                swarms.append(swarm_name)
            if swarm_name in ['HPSO-TVAC']:
                print(F'{i} delete', task_file)
                i += 1
                for d_file in task_file.glob('*'):
                    os.remove(d_file)
                task_file.rmdir()
    print(swarms)


if __name__ == '__main__':
    # task_des, task_result = get_task_by_type('top')
    # delete_all_task()
    delete_swarm_task()
    delete_not_single_evaluate_task()
    a = 1
