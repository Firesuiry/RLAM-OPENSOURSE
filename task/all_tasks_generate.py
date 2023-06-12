# 生成全流程任务
from matAgent.clpso import ClpsoSwarm
from matAgent.epso import EpsoSwarm
from matAgent.fdrpso import FdrpsoSwarm
from matAgent.hpso_tvac import HpsotvacSwarm
from matAgent.lips import LipsSwarm
from matAgent.olpso import OlpsoSwarm
from matAgent.pso import PsoSwarm
from matAgent.shpso import ShpsoSwarm
from matAgent.hrlepso_base import HrlepsoBaseSwarm
from matAgent.swarm.gwo import GwoSwarm
from matAgent.rlepso import RlepsoSwarm


def test_all_tasks_generate():
    evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, PsoSwarm, ShpsoSwarm, HrlepsoBaseSwarm,
                           GwoSwarm]
    base_evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, OlpsoSwarm, PsoSwarm, ShpsoSwarm,
                                EpsoSwarm, GwoSwarm]  # 都用一样的
    runtimes = 1
    separate_trains = [False]
    groups = [5, ]
    train_max_episode = 2
    train_max_steps = 100 * train_max_episode
    dims = [20, ]
    evaluate_function = list(range(1, 2, 1))

    task = {'type': 'top',
            'evaluate_optimizers': evaluate_optimizers,
            'base_evaluate_optimizers': base_evaluate_optimizers,
            'evaluate_function': evaluate_function,
            'runtimes': runtimes,
            'separate_trains': separate_trains,
            'groups': groups,
            'train_max_episode': train_max_episode,
            'train_max_steps': train_max_steps,
            'dims': dims,
            'train_times': 1,
            'lr_critic': 1e-4,
            'lr_actor': 1e-6,
            }

    return [task]


def all_tasks_generate():
    # evaluate_optimizers = [GwoSwarm]
    evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, PsoSwarm, ShpsoSwarm, HrlepsoBaseSwarm,
                           HpsotvacSwarm]
    # evaluate_optimizers = [PsoSwarm]
    base_evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, OlpsoSwarm, PsoSwarm, ShpsoSwarm,
                                EpsoSwarm, ]  # 都用一样的
    runtimes = 10
    separate_trains = [True, False]
    # groups = [1, 3, 5, 7, 9]
    groups = [5]
    train_max_episode = 400
    train_max_steps = train_max_episode * 100
    dims = [30,50]
    evaluate_function = list(range(1, 29, 1))

    task = {'type': 'top',
            'evaluate_optimizers': evaluate_optimizers,
            'base_evaluate_optimizers': base_evaluate_optimizers,
            'evaluate_function': evaluate_function,
            'runtimes': runtimes,
            'separate_trains': separate_trains,
            'groups': groups,
            'train_max_episode': train_max_episode,
            'train_max_steps': train_max_steps,
            'dims': dims,
            'train_times': 1,
            'lr_critic': 1e-4,
            'lr_actor': 1e-6,
            }

    return [task]


def CLPSO_tasks_generate():
    evaluate_optimizers = [ClpsoSwarm, ]
    base_evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, OlpsoSwarm, PsoSwarm, ShpsoSwarm,
                                EpsoSwarm]  # 都用一样的
    runtimes = 10
    # separate_trains = [True, False]
    separate_trains = [False]
    # groups = [1, 3, 5, 7, 9]
    groups = [5]
    train_max_episode = 600
    train_max_steps = train_max_episode * 100
    dims = [30]
    evaluate_function = list(range(1, 29, 1))

    task = {'type': 'top',
            'evaluate_optimizers': evaluate_optimizers,
            'base_evaluate_optimizers': base_evaluate_optimizers,
            'evaluate_function': evaluate_function,
            'runtimes': runtimes,
            'separate_trains': separate_trains,
            'groups': groups,
            'train_max_episode': train_max_episode,
            'train_max_steps': train_max_steps,
            'dims': dims,
            'train_times': 1,
            'lr_critic': 1e-3,
            'lr_actor': 1e-5,
            }

    return [task]


def QHrlepsoTrainTest():
    from matAgent.qrlepso.f16rlepso import F16Rlepso
    from matAgent.qrlepso.f64rlepso import F64Rlepso
    from matAgent.qrlepso.i8rlepso import I8Rlepso
    from matAgent.qrlepso.i16rlepso import I16Rlepso
    evaluate_optimizers = [F16Rlepso, F64Rlepso, I8Rlepso, I16Rlepso]
    # evaluate_optimizers = [I8Rlepso, ]
    base_evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, OlpsoSwarm, PsoSwarm, ShpsoSwarm,
                                EpsoSwarm]  # 都用一样的
    runtimes = 10
    separate_trains = [False]
    groups = [5, ]
    train_max_episode = 200
    train_max_steps = 20000
    dims = [30]
    evaluate_function = list(range(1, 29, 1))

    task = {'type': 'top',
            'evaluate_optimizers': evaluate_optimizers,
            'base_evaluate_optimizers': base_evaluate_optimizers,
            'evaluate_function': evaluate_function,
            'runtimes': runtimes,
            'separate_trains': separate_trains,
            'groups': groups,
            'train_max_episode': train_max_episode,
            'train_max_steps': train_max_steps,
            'dims': dims,
            'train_times': 1
            }

    return [task]


if __name__ == '__main__':
    test_all_tasks_generate()
