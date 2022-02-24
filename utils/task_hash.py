import copy
import hashlib
import json
import pathlib


def md5(s):
    m = hashlib.md5()
    b = s.encode(encoding='utf-8')
    m.update(b)
    str_md5 = m.hexdigest()
    return str_md5


def task2str(task):
    json_task = copy.deepcopy(task)

    def obj2str(obj):
        if type(obj) == type:
            return obj.optimizer_name
        elif type(obj) == list:
            for i in range(len(obj)):
                obj[i] = obj2str(obj[i])
        elif type(obj) == dict:
            for k, v in obj.items():
                obj[k] = obj2str(v)
        elif type(obj) in [int, float, tuple] or type(obj) is None:
            pass
        else:
            obj = str(obj)
        return obj

    json_task = obj2str(json_task)
    json_str = json.dumps(json_task)
    return json_str


def get_task_hash(task):
    task_str = task2str(task)
    task_md5 = md5(task_str)
    return task_md5


def get_task_hashs(tasks):
    return [get_task_hash(task) for task in tasks]
