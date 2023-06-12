import os
import time
from multiprocessing.dummy import freeze_support
from display.top_task_result_display import top_task_result_display
# from warnings import simplefilter
# simplefilter('error')
from task.evaluate_task_generate import generate_evaluate_tasks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing as mp

from task.all_tasks_generate import *
from task.tasks_run import task_run
from utils.task_hash import get_task_hash
from log import logger
import psutil

mem = psutil.virtual_memory()
task_progress = {}


def task_statistic(task, start=0, finish=0):
    global task_progress
    task_type = task.get('type')
    if task_type is None:
        return
    if task_type not in task_progress:
        task_progress[task_type] = {
            'all': 0,
            'finish': 0,
        }
    if start:
        task_progress[task_type]['all'] += 1
    if finish:
        task_progress[task_type]['finish'] += 1


def print_task_progress():
    global task_progress
    with open('progress.txt', 'w') as f:
        for k, v in task_progress.items():
            task_type = k
            all_num = v['all']
            finish_num = v['finish']
            info = f'{task_type} finish/all:{finish_num}/{all_num}\r\n'
            f.write(info)


def main(processes=1):
    if processes > mp.cpu_count():
        processes = mp.cpu_count()
    logger.info(f'main run at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    logger.info(f'processes:{processes}')
    need_run_tasks = []
    running_tasks = []
    wait_result_tasks = {}
    task_detail = {}
    # need_run_tasks += generate_evaluate_tasks()
    need_run_tasks += all_tasks_generate()
    # need_run_tasks += test_all_tasks_generate()

    mq = None
    pool = None
    run_epoch = 0
    if processes > 1:
        mq = mp.Manager().Queue(maxsize=500)
        pool = mp.Pool(processes=processes)
    while len(need_run_tasks) + len(running_tasks) > 0:
        run_epoch += 1
        if run_epoch % 10 == 0:
            print_task_progress()
            if mem.available < 10 * 1024 * 1024 * 1024:
                logger.info(f'free memory:{mem.available / 1024 / 1024 / 1024}G')
                if processes > 1:
                    pool.terminate()
                return 'restart'
        result = None
        if len(need_run_tasks) > 0:
            task = need_run_tasks.pop()
            running_task_md5 = get_task_hash(task)
            task_detail[running_task_md5] = task
            if running_task_md5 not in running_tasks:
                running_tasks.append(running_task_md5)
            task_statistic(task, start=1)
            if processes > 1:
                pool.apply_async(func=task_run, args=(task, mq,))
                logger.debug(f'添加多进程任务 {running_task_md5}')
            else:
                result = task_run(task)
        if processes > 1 and not mq.empty():
            logger.debug(f'从消息队列获得消息')
            result = mq.get()

        # 结果处理
        if result:
            # task_result = result['result']
            result_task_md5 = result['md5']
            other_need_tasks = result.get('needs')
            # logger.info(f"任务运行结果{result_task_md5}-{result}")
            logger.info(f"任务运行结束{result_task_md5}")
            if other_need_tasks:
                other_need_tasks_md5 = []
                for need_task in other_need_tasks:
                    other_need_tasks_md5.append(get_task_hash(need_task))
                    need_run_tasks.append(need_task)
                wait_result_tasks[result_task_md5] = other_need_tasks_md5
                logger.info(f"任务{result_task_md5}添加等待信息{wait_result_tasks[result_task_md5]}")
            else:
                if result_task_md5 in running_tasks:
                    running_tasks.remove(result_task_md5)
                task_statistic(result, finish=1)
                # 重启符合条件任务
                del_keys = []
                for restart_task_md5, needs in wait_result_tasks.items():
                    if result_task_md5 in needs:
                        needs.remove(result_task_md5)
                    if len(needs) == 0:
                        logger.info(f"条件满足 重启任务{restart_task_md5}")
                        del_keys.append(restart_task_md5)
                        need_run_tasks.append(task_detail[restart_task_md5])
                for del_key in del_keys:
                    del wait_result_tasks[del_key]
                if result['type'] == 'top':
                    top_task_result_display(result)
        else:
            if len(need_run_tasks) < 1:
                # logger.info('no need_run_task,wait')
                time.sleep(1)


import sys

if __name__ == '__main__':
    freeze_support()
    res = 'restart'
    while res == 'restart':
        res = main(32)
        # res = main(1)
        logger.info(f'main run finish res:{res}')
        time.sleep(60)
