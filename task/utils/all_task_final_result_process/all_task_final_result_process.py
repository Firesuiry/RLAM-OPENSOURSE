import pickle
from settings import *
from log import logger
import numpy as np


def all_task_final_result_process(task_result, evaluate_optimizer):
    # logger.debug(task_result)
    logger.debug('all_task_final_result_process')
    result = task_result['result']
    # logger.debug(result)

    train_ranks = []
    origin_ranks = []

    for f_num, f_result in result.items():
        logger.debug(f'函数:{f_num},res:{f_result}')
        for opt_name, opt_result in f_result.items():
            if 'train' in opt_name and evaluate_optimizer == opt_result['evaluate_optimizer']:
                train_ranks.append(get_rank(opt_name, opt_result, f_result))
            elif 'origin' in opt_name and evaluate_optimizer == opt_result['evaluate_optimizer']:
                origin_ranks.append(get_rank(opt_name, opt_result, f_result))

    train_average_rank = np.average(train_ranks)
    origin_average_rank = np.average(origin_ranks)
    # print(train_average_rank, train_ranks)
    # print(origin_average_rank, origin_ranks)
    return {
        'train_average_rank': train_average_rank,
        'origin_average_rank': origin_average_rank,
        # 'other_result': result,
    }


def get_rank(rank_opt_name, rank_opt_result, f_result):
    opt_ans = rank_opt_result['result'][-1][2]
    rank = 1
    for opt_name, opt_result in f_result.items():
        if opt_result['result'][-1][2] < opt_ans and 'origin' in opt_name:
            rank += 1
    return rank


if __name__ == '__main__':
    with open('all_task_final_result_process_test_input.pickle', 'rb') as f:
        data = pickle.load(f)
    all_task_final_result_process(data)
