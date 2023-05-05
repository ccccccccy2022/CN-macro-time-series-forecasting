# -*- coding = utf-8 -*-
# @time:2022/6/14 16:30
# Author:Tjd_T
# @File:model_train_base.py
# @Software:PyCharm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from model_tools.loss.dilate_loss import dilate_loss
import abc
import torch


class ModelTrainBase:
    def __init__(self, model, arg_dict):
        self.parameter_dict = arg_dict
        self.model = model
        self.opt = self.get_optimizer()
        self.lr_schedule = self.get_lr_schedule()

    def get_optimizer(self):
        opt = self.parameter_dict['optim']['name']
        if opt == 'SGD':
            my_opt = optim.SGD
        elif opt == 'ASGD':
            my_opt = optim.ASGD
        else:
            my_opt = optim.Adam(self.model.parameters(), self.parameter_dict['lr'], weight_decay=1e-3)
        return my_opt

    def get_lr_schedule(self):
        schedule = self.parameter_dict['lr_schedule']['name']
        my_lr_sch = None
        if schedule == 'Step':
            step_size = self.parameter_dict['lr_schedule']['step_size']
            gamma = self.parameter_dict['lr_schedule']['gama']
            my_lr_sch = lr_scheduler.StepLR(self.opt, step_size=step_size, gamma=gamma)
        return my_lr_sch

    def __lossfunc(self, loss_name, output, y):
        if loss_name == 'MSE':
            loss = F.mse_loss(output, y)
        elif loss_name == 'MAE':
            loss = F.l1_loss(output, y)
        elif loss_name == 'SmoothL1':
            loss = F.smooth_l1_loss(output, y)
        elif loss_name == 'DILATE':
            alpa = self.args['alpa']
            gamma = self.args['gamma']
            device = torch.device("cuda:0" if self.args['cuda'] else "cpu")
            loss = dilate_loss(output.unsqueeze(0), y.unsqueeze(0), alpa, gamma, device)
        else:
            loss = F.mse_loss(output, y)
            self.args['loss_func'][0] = 'MSE'
        return loss

    def compute_loss(self, predictions, labels):
        # 将预测和标记的维度展平, 防止出现维度不一致
        predictions = predictions.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
        loss = self.__lossfunc(self.parameter_dict['loss_func'][0], predictions, labels)
        return loss

    @abc.abstractmethod
    def train(self, data, epoch):
        pass

    @abc.abstractmethod
    def predict(self, data):
        pass


if __name__ == '__main__':
    pass
