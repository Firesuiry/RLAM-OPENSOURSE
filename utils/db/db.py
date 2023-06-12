from peewee import *
import json
import pickle

from settings import BASE_PATH
from utils.task_hash import task2str

sql_db_address = BASE_PATH + '/db.db'
print(sql_db_address)
db = SqliteDatabase(sql_db_address)


class Optimizer(Model):
    optimizer = CharField(max_length=255)
    train_result = TextField()
    train_result_binary = BlobField()
    dim = IntegerField()
    group = IntegerField()
    separate_train = BooleanField()
    max_fe = IntegerField()
    n_part = IntegerField()

    class Meta:
        database = db


Optimizer.create_table()


# db.create_tables([Optimizer])

def save_optimizer(results):
    for result in results:
        optimizer = result['optimizer']
        dim = result['dim']
        group = result['group']
        separate_train = result['separate_train']
        max_fe = int(result['max_fe'])
        n_part = int(result['n_part'])
        p = Optimizer.select().where((Optimizer.optimizer == optimizer) &
                                     (Optimizer.dim == dim) &
                                     (Optimizer.group == group) &
                                     (Optimizer.separate_train == separate_train) &
                                     (Optimizer.max_fe == max_fe) &
                                     (Optimizer.n_part == n_part)).first()
        if p is None:
            p = Optimizer(optimizer=optimizer, dim=dim, group=group, separate_train=separate_train, max_fe=max_fe,
                          n_part=n_part)
        p.train_result = task2str(result['train_result'])
        p.train_result_binary = pickle.dumps(result['train_result'])
        p.save()
        print('db save')


def get_optimizer_train_result(optimizer, dim, group, separate_train, max_fe, n_part):
    p = Optimizer.select().where((Optimizer.optimizer == optimizer) &
                                 (Optimizer.dim == int(dim)) &
                                 (Optimizer.group == int(group)) &
                                 (Optimizer.separate_train == separate_train) &
                                 (Optimizer.max_fe == int(max_fe)) &
                                 (Optimizer.n_part == int(n_part))).first()
    if p is None:
        return None
    else:
        data = p.train_result_binary
        return pickle.loads(p.train_result_binary)

# save_optimizer([{'optimizer': 'adam', 'dim': 2, 'group': 1, 'separate_train': True, 'max_fe': 10, 'n_part': 1,
#                  'train_result': {1: 1}}])
if __name__ == '__main__':
    p = Optimizer.select().first()
    data = p.train_result_binary
    print(data)
