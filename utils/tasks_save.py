import copy
import pickle
import json
import os

from evaluate.task_generate import generate_all_task
from settings import BASE_PATH


def save_tasks(tasks, task_name, path=BASE_PATH + '/data/task/'):
    json_tasks = copy.deepcopy(tasks)
    for i in range(len(tasks)):
        task = tasks[i]
        for k, v in task.items():
            print(k, v, type(v))
            if type(v) == type:
                json_tasks[i][k] = v.optimizer_name
            if type(v) == list:
                for j in range(len(v)):
                    json_tasks[i][k][j] = v[j].optimizer_name

    save_path = f'{path}{task_name}/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    json_str = json.dumps(json_tasks)
    with open(save_path + 'task.json', 'w') as f:
        f.write(json_str)

    with open(save_path + 'task.pickle', 'wb') as f:
        pickle.dump(tasks, f, 0)


if __name__ == '__main__':
    tasks = generate_all_task([20])
    print(tasks)
    save_tasks(tasks, 'test_task')
