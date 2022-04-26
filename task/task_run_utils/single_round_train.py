import copy
import os

from env.NormalEnv import NormalEnv
from settings import TASK_PATH
from task.task_run_utils.common import result_process
from train.ddpg import get_ddpg_object
from utils.task_hash import get_task_hash


def single_round_train_task_run(task, mq=None):
    assert task['type'] == 'single_round_train', '只接受类型为round_train的任务'
    optimizer = task['optimizer']
    train_max_episode = task['train_max_episode']
    train_max_steps = task['train_max_steps']
    fun_nums = task['fun_nums']
    group = task['group']
    max_fe = task['max_fe']
    dim = task['dim']
    n_part = task['n_part']
    round_id = task['round']
    lr_critic = task.get('lr_critic', 1e-7)
    lr_actor = task.get('lr_actor', 1e-9)
    # 训练过程
    gym_env = NormalEnv(obs_shape=(optimizer.obs_space,), action_shape=(optimizer.action_space * group,),
                        target_optimizer=optimizer, fun_nums=fun_nums, max_fe=max_fe, n_part=n_part, n_dim=dim)

    assert (gym_env.action_space.high == -gym_env.action_space.low)
    is_discrete = False
    task_md5 = get_task_hash(task)
    task_dir = TASK_PATH.joinpath(f'{task_md5}/')

    if not os.path.exists(task_dir.joinpath(f"ddpg_actor_final_round{round_id}.h5")):
        ddpg = get_ddpg_object(gym_env, discrete=is_discrete, memory_cap=10000000, lr_critic=lr_critic,
                               lr_actor=lr_actor)
        save_freq = train_max_episode / 20
        if save_freq < 1:
            save_freq = 1
        save_freq = int(save_freq)
        ddpg.train(max_episodes=train_max_episode, max_epochs=train_max_steps, task_path=task_dir, train_num=round_id,
                   save_freq=save_freq)

    all_models = list(task_dir.glob('ddpg_actor*.h5'))

    task_result = copy.deepcopy(task)
    task_result['result'] = all_models
    task_result['md5'] = get_task_hash(task)

    return result_process(task, task_result, mq)