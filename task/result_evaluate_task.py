import copy
import os
from pathlib import Path
import json
import numpy as np
import xlwt

from display.res_display import multi_res_display
from settings import BASE_PATH


def visible_res_generate(results: dict, path=BASE_PATH + '/data/result/'):
    # 创建一个Workbook对象，相当于创建了一个Excel文件
    if not os.path.exists(path):
        os.makedirs(path)
    path = Path(path)
    caches = {}

    for key, value in results.items():
        benchmark = key
        caches[benchmark] = {}

    for fun, optimizer_ress in results.items():
        benchmark = fun
        best_caches = []
        for optimizer, res in optimizer_ress.items():
            caches[benchmark][optimizer] = {'best': res['result'][-1][2], 'result': res['result']}
            best_caches.append(res['result'][-1][2])
        for optimizer, res in optimizer_ress.items():
            index = 0
            for i in range(len(best_caches)):
                if best_caches[i] < caches[benchmark][optimizer]['best']:
                    index += 1
            caches[benchmark][optimizer]['rank'] = index

    for benchmark, ress in caches.items():
        multi_res_display(ress, benchmark)

    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet('test01', cell_overwrite_ok=True)

    finish_hang = 0
    finish_lie = 0
    for benchmark, ress in caches.items():
        lie = 1
        for swarm_name, res in ress.items():
            sheet.write(0, lie, swarm_name)
            lie += 1
        finish_lie = lie
        for swarm_name, res in ress.items():
            sheet.write(0, lie, swarm_name)
            lie += 1
        break

    hang = 1

    for benchmark, ress in caches.items():
        sheet.write(hang, 0, '{}-best'.format(benchmark))
        finish_hang = hang + 1
        hang += 1

    hang = 1
    for benchmark, ress in caches.items():
        lie = 1
        for swarm_name, res in ress.items():
            sheet.write(hang, lie, float(str(res['best'])))
            sheet.write(hang, lie + finish_lie - 1, float(str(res['rank'])))
            lie += 1
        hang = hang + 1

    i = 0
    name = './data/data{}.xls'.format(i)
    while os.path.exists(name):
        i += 1
        name = './data/data{}.xls'.format(i)
    book.save(name)
    json_result = copy.deepcopy(results)
    for fun, optimizer_ress in json_result.items():
        for optimizer, res in optimizer_ress.items():
            json_result[fun][optimizer] = res['result'].tolist()
    print(json_result)
    json.dump(json_result, open('./data/data{}.json'.format(i), 'w'))
