import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import h5py
from pathlib import Path

from matAgent.baseAgent import sin_encode

if not os.path.exists('data/img/'):
    os.mkdir('data/img/')
def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称

        for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():  # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for name, d in g.items():  # 读取各层储存具体信息的Dataset类
                print("      {}: {}".format(name, d.value.shape))  # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
                print("      {}: {}".format(name, d.value))
    finally:
        f.close()


def plot(model, jinddu=10, title=None):
    print(title)
    # print(model.summary())
    x = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []

    for i in range(jinddu):
        # x.append(i)
        # res = model.predict([(i, 0)])
        x.append(i / jinddu)
        res = model.predict([(i / jinddu, 0)])
        y1.append(res[0][0])
        y2.append(res[0][1])
        y3.append(res[0][2])
        y4.append(res[0][3])
        y5.append(res[0][4])
        # print(i, model.predict([i / jinddu]))
    plt.plot(x, y1, x, y2, x, y3, x, y4, x, y5)
    plt.title(f'{title}')
    plt.savefig(f'data/img/{title}.png')
    plt.clf()
    # plt.show()


def task_print():
    model_fn = 'data/task/ac109973e3f51718a313ed9ba2cfa9eb/ddpg_actor_episode50_round0.h5'
    tasks = ['bbead8669f29fbc42c75d2487a3009d7', ]
    # model = tf.keras.models.load_model(model_fn)
    for task in tasks:
        path = Path(F'data/task/{task}/')
        i = 0
        for file in path.glob('ddpg_actor*.h5'):
            i += 1
            # if i % 5 != 0:
            #     continue
            model = tf.keras.models.load_model(str(file), custom_objects={'leaky_relu': tf.nn.leaky_relu})
            # print(model.summary())
            # print(model.predict([0]))
            plot(model, jinddu=30, title=f'{file.name}')
            # if i > 5:
            #     break


def show_process_model():
    model_fn = r'D:\paper\rlma\model\0406单层sin\ddpg_actor_episode120_round0.h5'

    model = tf.keras.models.load_model(str(model_fn), custom_objects={'leaky_relu': tf.nn.leaky_relu})

    xs = []
    ws = []
    c1s = []
    c2s = []
    multis = []
    for i in range(100):
        process = i / 50 - 1
        no_improve_fe = 0
        diversity = 0
        state = np.array(get_state(process, no_improve_fe, diversity))
        # print(state)
        actions = model.predict(state)
        w, other_coefficient, mutation_rate, multi = get_coefficients(actions[0])
        c1 = other_coefficient[0]
        c2 = other_coefficient[1]
        xs.append(i / 100)
        ws.append(w)
        c1s.append(c1)
        c2s.append(c2)
        multis.append(multi)
    plt.plot(xs, ws, xs, c1s, xs, c2s, xs, multis)
    plt.title(f'process')
    plt.savefig(f'data/img/process.png')
    plt.clf()


def show_model(model_fn=r'data/task/4727d73b85fd578c3caa307bfc042bd1/ddpg_actor_episode300_round0.h5'):
    # model_fn = r'D:\paper\rlma\model\0406单层sin\ddpg_actor_episode120_round0.h5'
    # model_fn = r'data/task/4727d73b85fd578c3caa307bfc042bd1/ddpg_actor_final_round0.h5'

    model = tf.keras.models.load_model(str(model_fn), custom_objects={'leaky_relu': tf.nn.leaky_relu})
    # print(model.summary())
    # print(model.layers[-2].activation)
    # print(model.get_layer('L5').outputs)
    # return
    # process = self.fe_num * 2 / self.fe_max - 1
    # no_improve_fe = (self.fe_num - self.last_best_update_fe) / self.fe_max
    # diversity = np.mean(np.std(self.xs, axis=0))

    # 进度关系图
    xs = []
    ws = []
    c1s = []
    c2s = []
    for i in range(100):
        process = i / 50 - 1
        no_improve_fe = 0
        diversity = 0
        state = np.array(get_state(process, no_improve_fe, diversity))
        # print(state)
        actions = model.predict(state)
        w, other_coefficient, mutation_rate = get_coefficients(actions[0])
        c1 = other_coefficient[0]
        c2 = other_coefficient[1]
        xs.append(i / 100)
        ws.append(w)
        c1s.append(c1)
        c2s.append(c2)
    plot_xwc(xs, ws, c1s, c2s, title=F'process-{model_fn.name}')
    # return
    # 未增长关系图
    xs = []
    ws = []
    c1s = []
    c2s = []
    for i in range(100):
        process = 0
        no_improve_fe = i / 100
        diversity = 0
        state = np.array(get_state(process, no_improve_fe, diversity))
        actions = model.predict([state])
        w, other_coefficient, mutation_rate = get_coefficients(actions[0])
        c1 = other_coefficient[0]
        c2 = other_coefficient[1]
        xs.append(i / 100)
        ws.append(w)
        c1s.append(c1)
        c2s.append(c2)
    plot_xwc(xs, ws, c1s, c2s, F'no_improve_fe-{model_fn.name}')

    # 进度关系图
    xs = []
    ws = []
    c1s = []
    c2s = []
    for i in range(100):
        process = 0
        no_improve_fe = 0
        diversity = i / 100
        state = np.array(get_state(process, no_improve_fe, diversity))
        actions = model.predict([state])
        w, other_coefficient, mutation_rate = get_coefficients(actions[0])
        c1 = other_coefficient[0]
        c2 = other_coefficient[1]
        xs.append(i / 100)
        ws.append(w)
        c1s.append(c1)
        c2s.append(c2)
    plot_xwc(xs, ws, c1s, c2s, F'diversity-{model_fn.name}')


def plot_xwc(xs, ws, c1s, c2s, title):
    plt.plot(xs, ws, label='w')
    plt.plot(xs, c1s, label='c1')
    plt.plot(xs, c2s, label='c2')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.legend()
    plt.title(f'{title}')
    plt.savefig(f'data/img/{title}.png')
    plt.clf()


def get_coefficients(actions, coefficients_multi=True, range_process=True):
    action = actions[0:5]
    multi_coefficient = action[-1] + 1
    mutation_rate = (action[-2] + 1) * 0.01
    if range_process:
        other_coefficient = action[1:-2] * 1.5 + 1.5
        w = action[0] * 0.4 + 0.5
        # w = action[0] * 10 + 0.5
    else:
        other_coefficient = action[1:-2]
        w = action[0]
    action_sum = np.sum(other_coefficient) + 1e-10
    if action_sum == 0:
        action_sum = 1e-10
    # if coefficients_multi:
    #     other_coefficient = other_coefficient / action_sum * multi_coefficient * 4
    return w, other_coefficient, mutation_rate


def get_state(process, no_improve_fe, diversity):
    next_state = [process, no_improve_fe, diversity]
    return [sin_encode(next_state, num=4)]


if __name__ == '__main__':
    # task_print()
    # show_process_model()
    # path = Path('data/task/f2d85fc221768d8e304030d4249bd365')
    path = Path(r'D:\develop\autoTrain\data\cache')
    for file in path.glob('ddpg_actor*.h5'):
        show_model(file)
