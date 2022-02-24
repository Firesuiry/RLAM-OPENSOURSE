from log import logger
import numpy as np


def top_task_result_display(result):
    logger.info('top_task_result_display')
    logger.info(result)
    evaluate_result = result['result']

    # tongji all improve
    train_average_ranks = []
    origin_average_ranks = []
    for res in evaluate_result:
        train_average_ranks.append(res['result'][0]['train_average_rank'])
        origin_average_ranks.append(res['result'][0]['origin_average_rank'])
    print(f'train_average_ranks:{np.mean(train_average_ranks)}')
    print(f'origin_average_ranks:{np.mean(origin_average_ranks)}')
    pass
