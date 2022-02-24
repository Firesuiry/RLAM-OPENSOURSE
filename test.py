import tensorflow as tf
import matplotlib.pyplot as plt

import h5py
from pathlib import Path


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
        x.append(i/jinddu)
        res = model.predict([(i/jinddu, 0)])
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


if __name__ == '__main__':
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
