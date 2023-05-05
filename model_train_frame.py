# -*- coding = utf-8 -*-
# @time:2022/6/14 16:11
# Author:Tjd_T
# @File:model_train_frame.py
# @Software:PyCharm
import copy
import json
import sklearn
import torch
import os
import shutil
from torch.utils.data import DataLoader
# from model_train_base import ModelTrainBsae
from model_tools.model_train_base import ModelTrainBase
from model_tools.bert.bert_model import BertModel
from model_tools.bert.bert_component import BertConfig
import numpy as np
import tqdm
# from torchsummary import summary
# from torchviz import make_dot
import os
import pickle
class ModelTrain(ModelTrainBase):
    def __init__(self, k, arg_dict):
        # 随机种子初始化
        if arg_dict['seed_flag']:
            torch.manual_seed(arg_dict['model_seed'])
        arg_dict['model_seed'] = torch.initial_seed()
        # 初始化模型
        bertconfig = BertConfig()
        model = BertModel(config=bertconfig, k=k, args=arg_dict, nnn_i=0, pn=0)
        # 初始化基础类，得到学习策略器和优化器
        super(ModelTrain, self).__init__(model, arg_dict)
        # 卷积的feature_map
        self.flag = False
        self.feature_map = list()
        self.register_cnn()
        # 位置编码
        self.positional_enc = self.init_positional_encoding(k)
        # 保存各个期的模型
        self.model_dict = dict()

    def init_positional_encoding(self, k):
        hidden_dim = 1 * (k + self.parameter_dict['categray_num'])
        max_seq_len = self.parameter_dict['max_seq_len']
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / hidden_dim) for i in range(hidden_dim)]
            if pos != 0 else np.zeros(hidden_dim) for pos in range(max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc ** 2, axis=1, keepdims=True))
        # 归一化
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        position_enc = torch.unsqueeze(position_enc, dim=0)
        return position_enc
    # def modeltorchviz(self, input, input2):
    #
    #     model=self.model
    #     # params: model = MSDNet(args).cuda()
    #     # params: input = (3, 32, 32)
    #     # params: input2 = torch.randn(1, 1, 28, 28).requires_grad_(True)  # 定义一个网络的输入值
    #     print(model)
    #     # summary(model, input)
    #     y = model(input2.cuda())  # 获取网络的预测值
    #     MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', input2)]))
    #     MyConvNetVis.format = "png"
    #     # 指定文件生成的文件夹
    #     # MyConvNetVis.directory = "data"
    #     # 生成文件
    #     MyConvNetVis.view()
    def train(self, data_loader, epoch):
        str_code = "train"
        data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%d" % (str_code, epoch), total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        categray_num = self.parameter_dict['categray_num']
        total_loss = 0
        self.model.train()
        i=0
        if len(data_iter)>0:
            for i, data in data_iter:
                x, y = data[0], data[1]
                data_input = x[:, :, :-categray_num - 3].float()
                if categray_num==0:
                    data_time = x[:, :, - 3:].long()
                else:
                    data_time = x[:, :, -categray_num - 3:-categray_num].long()
                data_categray = x[:, :, -categray_num:].long()
                y = y.float()
                try:
                    predictions = self.model(input_ids=data_categray, time_cate=data_time, dataset=data_input,
                                             positional_enc=self.positional_enc, labels=y
                                             )

                    # MyConvNetVis = make_dot(predictions,params=dict(self.model.named_parameters()))
                    # MyConvNetVis.format = "pdf"
                    # # 指定文件生成的文件夹
                    # # MyConvNetVis.directory = "data"
                    # # 生成文件
                    # MyConvNetVis.save('prediction')
                    # MyConvNetVis.view()
                    loss = self.compute_loss(predictions, y[:, self.parameter_dict['data_gap']])
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                except:break
            return total_loss / (i + 1)
        else:
            return total_loss

    def predict(self, data_loader, ynum):
        str_code = "predict"
        data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:" % (str_code,), total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        categray_num = self.parameter_dict['categray_num']
        nfor = self.parameter_dict['n_for']
        gap = self.parameter_dict['data_gap']
        result = list()
        self.model.eval()
        with torch.no_grad():
            for i, data in data_iter:
                x, y = data[0], data[1]
                data_input = x[:, :, :-categray_num - 3].float()
                if categray_num==0:
                    data_time = x[:, :, -categray_num - 3:].long()
                else:
                    data_time = x[:, :, -categray_num - 3:-categray_num].long()
                data_categray = x[:, :, -categray_num:].long()
                y = y.float()
                # 若注册了卷积输出，打开开关
                if self.parameter_dict['cnn_layer']:
                    self.flag = True
                # 加载模型
                #TODO 20220715
                self.model.load_state_dict(self.model_dict[str(ynum) + '_' + str(i)])
                # 预测结果
                predictions = self.model(input_ids=data_categray, time_cate=data_time, dataset=data_input,
                                         positional_enc=self.positional_enc, labels=y)

                predictions = predictions.detach().numpy().reshape(-1).tolist()
                #TODO 20221112
                result = result +predictions # 注意顺序是10,9,8,7...
                # 关掉输出开关
                if self.parameter_dict['cnn_layer']:
                    self.flag = False
        return result

    def train_method(self, data, train_size=1):
        from model_tools.bert_data import MyData, MyDataSet
        # from bert_data import MyData, MyDataSet
        nfor = self.parameter_dict['n_for']
        epoch = self.parameter_dict['epochs']
        ynums = self.parameter_dict['y_num']
        step = self.parameter_dict['data_gap']
        final_loss = dict()
        finnal_res = dict()
        finnal_rest=dict()
        # finnal_res = list()
        # 生成文件夹
        self.__mk_dir()
        # 准备dataset，dataload
        #TODO 20221106
        for minmax in [None]:
            finnal_rest[minmax]=list()
            finnal_res[minmax]=list()
            data_cls = MyData(data[2][0], data[3].values.flatten(), data[4][0], data[5].values.flatten(), data[1],minmax=minmax)
            train_total, train_sub = data_cls.get_train(0, train_size)
            vaild_total, vaild_sub = data_cls.get_vaild(0, train_size)
            test_total = data_cls.get_test()
            train_total_dataset = MyDataSet(train_total)  # torch的set自带的数据处理，只要传入可迭代对象
            train_total_dataload = DataLoader(dataset=train_total_dataset, batch_size=self.parameter_dict['batch_size'], drop_last=True)
            vaild_total_dataset = MyDataSet(vaild_total)  # torch的set自带的数据处理，只要传入可迭代对象
            vaild_total_dataload = DataLoader(dataset=vaild_total_dataset, batch_size=min(int(len(vaild_total)),self.parameter_dict['batch_size']), drop_last=True)

            # 训练# TODO 20220715 修改loss_list的位置，修改nfor和epoch的循环方式
            loss_list = list()
            for y_id in range(ynums + 1):
                threshold = 0
                for cur_nfor in range(0, nfor, -step):
                    for ep in range(epoch):
                        # TODO 20220715
                        if y_id ==0:
                            loss_v = self.train(train_total_dataload, ep)
                            loss_vv = self.train(vaild_total_dataload, ep)
                            loss_list.append(1/2*(loss_v+loss_vv))
                            self.model_dict[0] = copy.deepcopy(self.model.state_dict())
                        else:
                        #TODO 20220715
                            cur_sub,cur_sub_y = data_cls.get_train(cur_nfor, 1)
                            # cur_y_data = cur_sub_y[y_id - 1]
                            cur_y_dataset = MyDataSet(cur_sub[(y_id-1)*int(len(cur_sub)/len(cur_sub_y)):y_id*int(len(cur_sub)/len(cur_sub_y))])
                            cur_y_dataload = DataLoader(dataset=cur_y_dataset, batch_size=self.parameter_dict['batch_size'], drop_last=True,shuffle=False)
                            cur_y_loss = self.train(cur_y_dataload, ep)
                            cur_model = str(y_id) + '_' + str(cur_nfor)
                            if cur_y_loss<loss_list[-1]:
                                self.model_dict[cur_model] = copy.deepcopy(self.model.state_dict())
                            else:
                                self.model_dict[cur_model] = self.model_dict[0]
                            loss_list.append(cur_y_loss)
                        # 是否早停
                        threshold, can_stop = self.__early_stop(loss_list, ep, threshold)
                        if can_stop:
                            break
                final_loss['y_id_{}_loss'.format(y_id)] = loss_list
                if y_id > 0:
                    cur_predict_data = test_total[y_id - 1]
                    cur_predict_dataset = MyDataSet(cur_predict_data)
                    cur_predict_dataload = DataLoader(dataset=cur_predict_dataset, batch_size=1)
                    result = self.predict(cur_predict_dataload, y_id)
                    if minmax==None:
                        _, cur_sub = data_cls.get_train(nfor-1, 1)
                        cur_y_data = cur_sub[y_id - 1]
                        for x in range(len(cur_y_data)-nfor):
                            if x==0:
                                cur_y_datasett = MyDataSet(cur_y_data[-nfor:])
                            else:
                                cur_y_datasett = MyDataSet(cur_y_data[-nfor-x:-x])
                            cur_predict_dataloadtest = DataLoader(dataset=cur_y_datasett, batch_size=1)
                            # resultt = self.predict(cur_predict_dataloadtest, y_id)
                            # finnal_rest[minmax].append(resultt)
                    finnal_res[minmax].append(result)
        # 模型和数据的保存工作
        self.save_file(final_loss, data)
        # fity = self.__fitpredicty
        finnal_res = self.revise_value(finnal_res, data[6])#,datalabel=[None,'minmax','min']
        return finnal_res,1/2*(loss_v+loss_vv)

    def save_file(self, total_loss, data):
        # 1. 日志输出
        log_path = self.parameter_dict['trans_former_save'] + '/log/'
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        with open(log_path+'train_loss.txt','w') as f:
            json.dump(total_loss,f)
        # 2.跑模型的数据输出
        if not os.path.exists(self.parameter_dict['trans_former_save'] + '/model'):
            os.makedirs(self.parameter_dict['trans_former_save'] + '/model')
        model_path = self.parameter_dict['trans_former_save'] + '/model' + "/"
        if os.path.exists(model_path + 'data_and_parameter.pkl'):
            num=len(os.listdir(model_path))
            with open(model_path + 'data_and_parameter'+str(num)+'.pkl', 'wb') as pk_file:
                pickle.dump(data, pk_file)
        else:
            with open(model_path + 'data_and_parameter.pkl', 'wb') as pk_file:
                pickle.dump(data, pk_file)
        # 3.增加model的一次性输出，只在最后的滚动期输出模型的保存参数
        if self.parameter_dict['output_model']:
            # 开始保存
            with open(model_path + 'model_dict' + '.pkl', 'wb') as pk_file:
                pickle.dump(self.model_dict, pk_file)
  #TODO 20220715 修改true_y【：-nfor,col]为truey[:,col]
    def revise_value(self, data, true_y,datalabel=None):
        if datalabel==None:
            data=data[datalabel]
            data = np.array(data).T
            true_y = true_y.values
            ynum = self.parameter_dict['y_num']
            nfor = self.parameter_dict['n_for']
            for col in range(ynum):
                true_y_value = true_y[:, col]
                for row in range(nfor):
                    cur_val = np.array(data[row, col]).reshape((-1, 1))
                    data[row, col] = self.__normalization_revise(true_y_value, cur_val)[0]
            return data
        else:
            # assert type(datalabel) == list
            true_y = true_y.values
            Data=data[None]
            ynum = self.parameter_dict['y_num']
            nfor = self.parameter_dict['n_for']
            try:
                minmax=data['minmax']
                min=data['min']
                minmax =np.array(minmax).T
                min =np.array(min).T

                Data = np.array(Data).T

                ymaxmin = true_y.max() - true_y.min()
                ymin = true_y.min()
                data0=(Data-min)*minmax+min
                data0=ymaxmin*(data0-true_y.mean(axis=0))+ymin
                return data0
            except:
                for col in range(ynum):
                    true_y_value = true_y[:, col]
                    for row in range(nfor):
                        cur_val = np.array(Data[row, col]).reshape((-1, 1))
                        Data[row, col] = self.__normalization_revise(true_y_value, cur_val)[0]
                return Data

    def register_cnn(self):
        if self.parameter_dict['cnn_layer']:
            for i in range(self.parameter_dict['cnn_nums']):
                self.model.reembed.datasetembedding.data_embeddings.cnn_list[i].register_forward_hook(
                    self.__get_feature)

    def __mk_dir(self):
        try:
            shutil.rmtree(self.parameter_dict['trans_former_save'] + '/dict' + str(self.parameter_dict['PN']))
            os.mkdir(self.parameter_dict['trans_former_save'] + '/dict' + str(self.parameter_dict['PN']))
        except:
            os.mkdir(self.parameter_dict['trans_former_save'] + '/dict' + str(self.parameter_dict['PN']))

    def __early_stop(self, loss_list, ep, threshold):
        patient = self.parameter_dict['patience']
        min_loss = min(loss_list)
        if loss_list[-1] > min_loss:
            threshold += 1
            # TODO:待优化自定义
            self.parameter_dict['lr'] *= 0.6
            self.opt = self.get_optimizer()
        else:
            threshold = 0
        if threshold >= patient:
            print("epoch {} has the lowest loss".format(ep))
            print("early stop!")
            return threshold, True
        return threshold, False

    def __normalization_revise(self, ori_data, target):  # x,1 dimension np.array
        y_scaler = None
        method = self.parameter_dict['y_normalize']
        if method == 'z-score':
            y_scaler = sklearn.preprocessing.StandardScaler().fit(ori_data.reshape(-1, 1))
        elif method == '0-1':
            y_scaler = sklearn.preprocessing.MinMaxScaler().fit(ori_data.reshape(-1, 1))
        elif method == 'MaxAbsScaler':
            y_scaler = sklearn.preprocessing.MaxAbsScaler().fit(ori_data.reshape(-1, 1))
        try:
            y = y_scaler.inverse_transform(target).flatten()
        except:
            y = y_scaler.inverse_transform(target,1).flatten()

        return y

    def __get_feature(self, model, datainput, dataout):
        if self.flag:
            tem = list()
            tem.append(datainput[0].detach().numpy().flatten())
            tem.append(dataout.detach().numpy().flatten())
            self.feature_map.append(tem)


if __name__ == '__main__':
    pass
