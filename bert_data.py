# -*- coding = utf-8 -*-
# @time:2022/6/9 14:08
# Author:Tjd_T
# @File:bert_data.py
# @Software:PyCharm
import copy
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


# TODO 20221106
class MyData:
    def __init__(self, train_x, train_y, test_x, test_y, arg_dict,minmax=None):
        self.args = arg_dict
        if minmax == None:
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y
        elif minmax == 'minmax':
            y_num = self.args['y_num']
            n_for=self.args['n_for']
            length = int(train_x.shape[0] / y_num)
            mmtrainx=[]
            mmtrainy=[]
            mmtestx=[]
            for y_id in range(y_num):
                cur_start = y_id * length
                mmtrainx=mmtrainx+[train_x.iloc[cur_start:cur_start + x+1].max()-train_x.iloc[cur_start:cur_start + x+1].min() for x in range(length)]
                mmtrainy=mmtrainy+[train_y[cur_start:cur_start + x+1].max()-train_y[cur_start:cur_start + x+1].min() for x in range(length)]
                for x in range(n_for):
                    a=[]
                    for j in range(train_x.shape[1]):
                        a.append(max(train_x.iloc[cur_start:cur_start + length, j].max(),
                                      test_x.iloc[cur_start:cur_start + x + 1, j].max()) - min(
                            train_x.iloc[cur_start:cur_start + length, j].min(),
                            test_x.iloc[cur_start:cur_start + x + 1, j].min()))
                    mmtestx=mmtestx+[pd.Series(a,index=train_x.columns)]

            self.train_x=pd.DataFrame(np.array(mmtrainx),index=train_x.index,columns=train_x.columns)
            self.test_x=pd.DataFrame(np.array(mmtestx),index=test_x.index,columns=test_x.columns)
            self.train_y=np.array(mmtrainy)
        else:
            y_num = self.args['y_num']
            n_for = self.args['n_for']
            length = int(train_x.shape[0] / y_num)
            mmtrainx = []
            mmtrainy = []
            mmtestx = []
            for y_id in range(y_num):
                cur_start = y_id * length
                mmtrainx = mmtrainx + [
                    train_x.iloc[cur_start:cur_start + x + 1].min()
                    for x in range(length)]
                mmtrainy = mmtrainy + [train_y[cur_start:cur_start + x + 1].min() for x in
                    range(length)]
                for x in range(n_for):
                    a = []
                    for j in range(train_x.shape[1]):
                        a.append(min(
                            train_x.iloc[cur_start:cur_start + length, j].min(),
                            test_x.iloc[cur_start:cur_start + x + 1, j].min()))
                    mmtestx = mmtestx + [pd.Series(a, index=train_x.columns)]

            self.train_x = pd.DataFrame(np.array(mmtrainx), index=train_x.index, columns=train_x.columns)
            self.test_x = pd.DataFrame(np.array(mmtestx), index=test_x.index, columns=test_x.columns)
            self.train_y = np.array(mmtrainy)

    def __mask(self, data, cur_yid):
        y_num = self.args['y_num']
        for i in range(y_num):
            if i != cur_yid:
                data[:, i] = 0
        return data

    def __split_train_data(self, data, nfor):
        data_x, data_y,test_x = data[0], data[1],data[2]
        length = int(data_x.shape[0] / self.args['y_num'])
        max_length = self.args['max_seq_len']
        if max_length > length:
            raise ValueError('max_len大于当前数据集长度')
        n_for1 = self.args['n_for']
        step = self.args['batch_step']
        train_total_data = list()
        train_sub_data = list()
        for y_id in range(self.args['y_num']):
            start = y_id * length
            end = start + length +(n_for1) - max_length - (nfor+1)+1 #预测期越长，end结束的点越靠前，减去的数量越多,-（nfor+1）最后一片留给预测预测一期，最后2片留给预测2期
            meta_data = copy.deepcopy(data_x)
            meta_data = meta_data.values
            meta_data = self.__mask(meta_data, y_id)
            test_data=self.__mask(test_x.values, y_id)
            newmeta_data = np.vstack([meta_data[start:start+length],test_data[y_id*n_for1:(y_id+1)*n_for1]])
            cur_y_train_data = list()
            for cur_start in range(start, end, step):
                cur_x_chip = newmeta_data[n_for1-(nfor+1)+cur_start-start:n_for1-(nfor+1)+cur_start-start + max_length, :]
                cur_y_chip = data_y[cur_start:cur_start + max_length]
                if len(cur_x_chip)<max_length or len(cur_y_chip)<max_length:
                    pass
                else:
                    cur_chip = (cur_x_chip, cur_y_chip)
                    cur_y_train_data.append(cur_y_chip)
                    train_total_data.append(cur_chip)
            train_sub_data.append(cur_y_train_data)
        return train_total_data, train_sub_data
    def __split_predict(self, data, n_for, is_tests=False):
        x_data, y_data = data[0], data[1]
        x_data = x_data.values
        max_length = self.args['max_seq_len']
        predict_data = list()
        for i in range(n_for):
            cur_x_chips = x_data[i:i + max_length, :]
            if is_tests:
                cur_y_chips = y_data
            else:
                cur_y_chips = y_data[i:i + max_length]
            cur_chips = (cur_x_chips, cur_y_chips)
            predict_data.append(cur_chips)
        return predict_data

    def get_train(self, nfor, datasize=1):
        tempt_x_data = list()
        tempt_y_data = list()
        y_num = self.args['y_num']
        length = int(self.train_x.shape[0] / y_num)
        train_size = int(length * datasize)
        for y_id in range(y_num):
            cur_end = (y_id+1) * length
            tempt_x_data.append(self.train_x.iloc[cur_end-train_size:cur_end, :])
            tempt_y_data.append(self.train_y[cur_end-train_size:cur_end])
        tempt_x = pd.concat(tempt_x_data, axis=0)
        tempt_y = np.hstack(tempt_y_data)
        data = [tempt_x, tempt_y,self.test_x]
        return self.__split_train_data(data, nfor)

    def get_vaild(self, nfor,datasize):
        tempv_x_data = list()
        tempv_y_data = list()
        y_num = self.args['y_num']
        n_for1=self.args['n_for']
        max_length = self.args['max_seq_len']
        length = int(self.train_x.shape[0] / y_num)
        train_size = max(max_length,int(length *(1- datasize)))
        for y_id in range(y_num):
            cur_train_start = y_id * length
            cur_chip = (self.train_x.iloc[cur_train_start+(n_for1-(nfor+1)):cur_train_start+max_length+(n_for1-(nfor+1)), :].values,self.train_y[cur_train_start:cur_train_start+max_length]
                        )
            tempv_x_data.append(cur_chip)
            tempv_y_data.append([self.train_y[cur_train_start:cur_train_start+max_length]])
        return tempv_x_data,tempv_y_data

    def get_test(self):
        y_num = self.args['y_num']
        max_length = self.args['max_seq_len']
        length = int(self.train_x.shape[0] / y_num)
        n_for = self.args['n_for']
        predict_data = list()
        for y_id in range(y_num):
            cur_train_start = (y_id + 1) * length - max_length
            cur_test_start = y_id * n_for
            cur_train_x = self.train_x.iloc[cur_train_start:cur_train_start + max_length, :]
            cur_test_x = self.test_x.iloc[cur_test_start:cur_test_start + n_for]
            cur_y = self.train_y[cur_train_start:cur_train_start + max_length]
            cur_x = pd.concat([cur_train_x, cur_test_x], axis=0)
            data = (cur_x, cur_y)
            predict_data.append(self.__split_predict(data, n_for, True))
        return predict_data


class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x_data, label = self.data[index][0], self.data[index][1]
        return x_data, label

    def __len__(self):
        return len(self.data)
