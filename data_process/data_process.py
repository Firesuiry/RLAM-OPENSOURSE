import copy
import json
import numpy as np

from display.res_display import multi_res_display


def data_process(data):
    rank(data)


def rank(data):
    rank_data = {}
    swarm_rank_cache = {}
    swarm_rank = {}

    for dim, dim_data in data.items():
        rank_data[dim] = {}
        for f_num, fun_data in dim_data.items():
            rank_data[dim][f_num] = {}
            for swarm_name, swarm_data in fun_data.items():
                swarm_rank_cache[swarm_name] = []
                swarm_rank[swarm_name] = []
                # fun_data是结果的列表
                best_values = []
                for task_record in swarm_data:
                    best_values.append(task_record['record'][-1][2])
                avarage_best_value = np.average(best_values)
                rank_data[dim][f_num][swarm_name] = avarage_best_value
    rank = copy.deepcopy(rank_data)
    for dim, dim_data in rank_data.items():
        for f_num, fun_data in dim_data.items():
            for swarm_name, avarage_best_value in fun_data.items():
                rank[dim][f_num][swarm_name] = 1
                for avarage_best_value2 in fun_data.values():
                    if avarage_best_value2 < avarage_best_value:
                        rank[dim][f_num][swarm_name] += 1
                swarm_rank_cache[swarm_name].append(rank[dim][f_num][swarm_name])
    for swarm_name in swarm_rank.keys():
        swarm_rank[swarm_name] = np.average(swarm_rank_cache[swarm_name])

    print(swarm_rank_cache)
    print(swarm_rank)
    swarm_rank = sorted(swarm_rank.items(), key=lambda kv: (kv[1], kv[0]))
    for k, v in swarm_rank:
        print(f'算法：{k: <12} 排名：{round(v,2)}')


def display(data, path):
    for dim, dim_data in data.items():
        for f_num, fun_data in dim_data.items():
            benchmark = f'F{f_num}-{dim}'
            ress = {}

            for swarm_name, swarm_data in fun_data.items():
                cache = []
                for task_record in swarm_data:
                    cache.append(task_record['record'])
                cache = np.array(cache)
                cache = np.average(cache, axis=0)
                ress[swarm_name] = cache
            multi_res_display(ress, benchmark, path=path)


def path_process(path):
    with open(path + 'json.json', 'r') as f:
        data_str = f.read()
    data = json.loads(data_str)
    # display(data, path)
    rank(data)



if __name__ == '__main__':
    path = r'D:/develop/swam/hard_rlepso/task_result/0a1e22bed84af530be5e4d435e834bcc/'
    path_process(path)
