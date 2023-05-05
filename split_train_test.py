# -*- coding = utf-8 -*-
# @time:2022/5/26 10:43
# Author:Tjd_T
# @File:split_train_test.py
# @Software:PyCharm
import numpy as np
import pandas as pd
import datetime
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def train_test_data_split(data_orgin, arg_dict):
    # 以下全是参数
    nnn = arg_dict['nnn']
    train_x, train_y, test_x, test_y, time = [], [], [], [], []
    k = arg_dict['k_step']
    y = arg_dict['y']
    n_for = arg_dict['n_for']
    # 修改了让数据起始位点不更新
    first_roll_start = arg_dict['n_ori_start']
    cur_roll_start = arg_dict['n_start']
    n_in = arg_dict['n_in']
    n_roll = [0] * len(arg_dict['n_roll'])
    length = arg_dict['length']
    # y
    ya = np.array(y['VAS'].values)
    ya[np.isnan(ya)] = 0
    yatime = np.array(y['time'])

    dataa2 = data_orgin
    ya22 = ya
    for j in range(0, nnn, 1):
        print('n_start is %s' % first_roll_start)
        if j % k == 0:
            train_x.append(dataa2.iloc[int(n_roll[j] + first_roll_start):int(n_roll[j] + length + cur_roll_start), :])
            train_y.append(
                ya22[int(n_in + n_roll[j] + first_roll_start):int(length + n_in + n_roll[j] + cur_roll_start)])
        else:
            train_x[j] = train_x[j - 1]
            train_y[j] = train_y[j - 1]
        test_x.append(
            dataa2.iloc[int(length + n_roll[j] + cur_roll_start):int(length + n_for + n_roll[j] + cur_roll_start), :])
        test_y.append(ya22[int(length + n_in + n_roll[j] + cur_roll_start):int(
            length + n_in + n_for + n_roll[j] + cur_roll_start)])
        time.append(yatime[int(length + n_in + n_roll[j] + cur_roll_start):int(
            length + n_in + n_for + n_roll[j] + cur_roll_start)])
    return test_y, time, train_x, train_y, test_x


def trans_std_flag(arg_dict):
    """
    为了兼容过去版本和新版本，将对x的标准化转化为一个参数
    @param arg_dict: 参数字典
    @return:
    """
    try:
        x_std_flag = arg_dict['x_std_flag']
    except KeyError:
        normalization_flag = arg_dict['normalization_flag']
        min_max_flag = arg_dict['min_max_flag']
        if normalization_flag == 1:
            x_std_flag = 1
        else:
            if min_max_flag == 2 or min_max_flag == 3:
                x_std_flag = 2
            else:
                x_std_flag = 0

    return x_std_flag


def train_data_normalize(train_x, test_x, arg_dict):
    n_for = arg_dict['n_for']
    nnn = arg_dict['nnn']
    length = arg_dict['length']
    x_std_flag = trans_std_flag(arg_dict)
    stestx, strainx, trainadd_x, strainadd_x = [], [], [], []
    for j in range(nnn):
        trainadd_x.append(np.vstack([train_x[j], test_x[j]]))
        if x_std_flag == 1:  # 方差标准化
            if (np.std(train_x[j], axis=0) == 0).any():
                strainadd_x.append(trainadd_x[j])
            else:
                strainadd_x.append(StandardScaler().fit(trainadd_x[j][:length]).transformer(trainadd_x[j]))
            strainx.append(pd.DataFrame(strainadd_x[j][:-n_for], columns=train_x[j].columns))
            stestx.append(pd.DataFrame(strainadd_x[j][-n_for:], columns=train_x[j].columns))
        elif x_std_flag == 2:  # 最大最小标准化
            strainadd_x.append(MinMaxScaler().fit(trainadd_x[j][:length]).transform(trainadd_x[j]))
            del_column = train_x[j].columns[np.isnan(strainadd_x[j]).any(axis=0)]
            strainx.append(pd.DataFrame(strainadd_x[j][:-int(n_for)], columns=train_x[j].columns))
            strainx[j].drop(del_column, axis=1, inplace=True)
            stestx.append(pd.DataFrame(strainadd_x[j][-int(n_for):], columns=train_x[j].columns))
            stestx[j].drop(del_column, axis=1, inplace=True)
        else:  # 不做标准化
            strainadd_x.append(trainadd_x[j].copy())
            strainx.append(pd.DataFrame(strainadd_x[j][:-n_for], columns=train_x[j].columns))
            stestx.append(pd.DataFrame(strainadd_x[j][-n_for:], columns=train_x[j].columns))
    return strainx, stestx


def get_train_test(arg_dict):
    data_save = arg_dict['data_save']
    with open(data_save, 'rb') as fd:
        dataa = pickle.load(fd)
    # 切分数据集
    test_y, time, train_x, train_y, test_x = train_test_data_split(dataa, arg_dict)
    # 标准化x
    strainx, stestx = train_data_normalize(train_x, test_x, arg_dict, )
    print('^' * 10 + '标准化完成' + '^' * 10)
    return train_y, strainx, test_y, stestx, train_x


if __name__ == '__main__':
    pass
