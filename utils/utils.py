def set_dict(dic, keys, value):
    for key in keys[:-1]:
        if key in dic:
            dic = dic[key]
        else:
            dic[key] = {}
            dic = dic[key]
    key = keys[-1]
    if key in dic:
        pass
    else:
        dic[key] = []
    dic[key].append(value)
