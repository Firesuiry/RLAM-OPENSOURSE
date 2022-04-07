import matplotlib.pyplot as plt
import json
import numpy as np
import math



MIN_MIN_NUM = 1.97e-308



def single_res_display(res, title='0'):
    print(res)
    x_data = []
    mean_data = []
    best_data = []
    num = 0
    for data in res:
        num += 1
        x_data.append(num)
        mean_data.append(math.log(data[0] + MIN_MIN_NUM, 10))
        best_data.append(math.log(data[1] + MIN_MIN_NUM, 10))

    plt.plot(x_data, mean_data, color='red', linewidth=2.0, linestyle='--')
    plt.plot(x_data, best_data, color='blue', linewidth=3.0, linestyle='-.')
    plt.title(title)
    # plt.show()
    plt.savefig('data/{}.png'.format(title))
    plt.close()
    json_str = json.dumps(res)
    with open('data/{}.json0'.format(title), 'w', encoding='utf-8') as f:
        f.write(json_str)


colors = ['#00FFFF', 'g', 'c', 'm', 'y', 'k', '#FFFF00', '#0f0f0f', '#672304', 'red']


def multi_res_display(ress, orititle='', path='data/img/'):
    title = '000-{}-对比图'.format(orititle)
    print(title)
    style = ['-.', '--', ':']

    i = 0
    for swarm_name, res in ress.items():
        plot_data = []
        x_data = []
        # if 'lips' in swarm_name.lower():
        #     continue
        # print(swarm_name, orititle)
        # print(res, np.array(res).shape)
        # print(res[0][1])
        for data in res['result']:
            x_data.append(data[0])
            plot_data.append(data[2])
        # min_value = np.min(plot_data) + 10000
        # plot_data = np.array(plot_data)
        # plot_data[plot_data > 10000 + min_value] = min_value + 10000
        plt.plot(x_data, plot_data, color=colors[i % len(colors)], linewidth=2.0, linestyle=style[i % len(style)],
                 label='{}'.format(swarm_name.upper().replace('SWARM', '')), scaley=True)
        i += 1

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.legend()
    plt.title('{}'.format(orititle))
    plt.ylabel('最优值（以10为底对数）')
    plt.xlabel('迭代次数')
    plt.savefig(f'{path}{title}.png')
    plt.close()
