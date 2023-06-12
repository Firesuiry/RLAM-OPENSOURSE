from matAgent.awpso import AwpsoSwarm
from matAgent.clpso import ClpsoSwarm
from matAgent.epso import EpsoSwarm
from matAgent.fdrpso import FdrpsoSwarm
from matAgent.hpso_tvac import HpsotvacSwarm
from matAgent.lips import LipsSwarm
from matAgent.olpso import OlpsoSwarm
from matAgent.pppso import PppsoSwarm
from matAgent.pso import PsoSwarm
from matAgent.shpso import ShpsoSwarm
from matAgent.hrlepso_base import HrlepsoBaseSwarm
from matAgent.swarm.gwo import GwoSwarm

from matAgent.adaptionPso.f1pso import FT1PsoSwarm
from matAgent.adaptionPso.f2pso import FT2PsoSwarm
from matAgent.adaptionPso.success_history_pso import SuccessHistoryPsoSwarm
from matAgent.adaptionPso.qlpso import QlpsoSwarm

from utils.db.db import get_optimizer_train_result

new_result_evaluate_task_test_dic = {
    'type': 'new_result_evaluate',
    'optimizer_model_list': [
        {
            'optimizer': 'optimizer_class',
            'fun_model': {
                1: [
                    'model_path1',
                ]
            }
        }
    ],
    'evaluate_function': list(range(1, 29, 1)),
    'group': 1,
    'max_fe': 1e4,
    'n_part': 100,
    'dim': 20,
    'runtimes': 10,
}

funs = list(range(1, 29, 1))
no_model_fun_model = {}
for fun in funs:
    no_model_fun_model[fun] = [None]
dim = 30
runtimes = 10
max_fe = 1e4
group = 5
separate_train = False
n_part = 100


def generate_evaluate_tasks():
    # no_train_optimizers = [FT1PsoSwarm, FT2PsoSwarm, SuccessHistoryPsoSwarm, QlpsoSwarm]
    train_optimizers = [PsoSwarm, ]
    no_train_optimizers = [PsoSwarm, ]

    # no_train_optimizers = [ClpsoSwarm, FdrpsoSwarm, LipsSwarm, PsoSwarm, ShpsoSwarm, AwpsoSwarm, PppsoSwarm, EpsoSwarm]
    # train_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, PsoSwarm, ShpsoSwarm, ]
    optimizer_model_list = []

    for no_train_optimizer in no_train_optimizers:
        optimizer_model_list.append({
            'optimizer': no_train_optimizer,
            'fun_model': no_model_fun_model,
        })
    for train_optimizer in train_optimizers:
        fun_model = get_optimizer_train_result(train_optimizer.optimizer_name, dim, group, separate_train, max_fe,
                                               n_part)
        fun_model = no_model_fun_model if fun_model is None else fun_model
        optimizer_model_list.append({
            'optimizer': train_optimizer,
            'fun_model': fun_model,
        })

    new_result_evaluate_task_dic = {
        'type': 'new_result_evaluate',
        'optimizer_model_list': optimizer_model_list,
        'evaluate_function': list(range(1, 29, 1)),
        'group': group,
        'max_fe': max_fe,
        'n_part': n_part,
        'dim': dim,
        'runtimes': runtimes,
    }

    return [new_result_evaluate_task_dic]


if __name__ == '__main__':
    a = generate_evaluate_tasks()
    print(a)
    pass
