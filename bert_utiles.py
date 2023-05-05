# -*- coding = utf-8 -*-
# @time:2021/11/26 16:19
# Author:Tjd_T
# @File:bert_utiles.py
# @Software:PyCharm

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from model_tools.models.model import Informer
from model_tools.bert.bert_component import BertLayerNorm


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class graph_constructor(nn.Module):
    def __init__(self, nnodes, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes  # max_seq_len
        self.emb1 = nn.Embedding(nnodes, dim)  # dim=1+kfilter or 1+args['dim']
        self.emb2 = nn.Embedding(nnodes, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

        self.k = int(1 / 2 * nnodes)
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.device = device
        self.idx = torch.arange(self.nnodes).to(self.device)

    def forward(self):
        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.selu(torch.tanh(self.alpha * a))
        mask = torch.zeros(self.idx.size(0), self.idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)  # return排序后的value,indice
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)  # 按axis=1轴标准化
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        try:
            w[:, 1::2] = torch.cos(position * div_term)
        except:
            w[:, 1::2] = torch.cos(position * div_term)[:, :-1]

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()
        # year_size=20
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        # self.year_embed = Embed(year_size,d_model)

    def forward(self, x):
        x = x.long()
        day_x = self.day_embed(x[:, :, -1])
        month_x = self.month_embed(x[:, :, -2])
        # year_x=self.year_embed(x[:,:,-3])
        return day_x + month_x


class EMNet(nn.Module):
    def __init__(self, config, k, args):
        super(EMNet, self).__init__()
        self.config = config
        self.k = k
        self.args = args
        self.selu = F.selu
        if not args['cnn_layer']:  # 不使用卷积层
            modules_body = []
            self.l1 = nn.Linear(int(args['max_seq_len'] * (k - 3)), int(args['max_seq_len']))
            self.l3 = nn.Linear(args['max_seq_len'], int(args['max_seq_len'] * (config.func(k, args)[0])))
            for i in range(10):
                modules_body.append(
                    nn.Linear(int(args['max_seq_len']), int(args['max_seq_len'])))
            self.body = nn.Sequential(*modules_body)
        else:
            self.cnn_list = nn.ModuleList()
            self.l1 = nn.Linear(int(args['max_seq_len'] * (k - 3)), int(args['max_seq_len']))
            self.l3 = nn.Linear(args['max_seq_len'], int(args['max_seq_len'] * (config.func(k, args)[0])))
            for i in range(self.args['cnn_nums']):
                self.cnn_list.append(
                    nn.Conv1d(1, 1, kernel_size=(3,), padding=True))

    def forward(self, dataset):
        if not self.args['cnn_layer']:
            x = dataset.view(-1, int(dataset.shape[1] * dataset.shape[2]))
            x = self.l1(x)
            res = self.body(self.selu(x))
            res += x
            return self.selu(self.l3(res)).reshape(-1, self.args['max_seq_len'], self.config.func(self.k, self.args)[0])
        else:
            x_list = []
            for i in range(self.args['cnn_nums']):
                x_tem = dataset[:, :, i]
                x_tem = x_tem.view(1, 1, -1)
                x_tem = self.selu(self.cnn_list[i](x_tem)).reshape(dataset.shape[0], -1)
                x_list.append(x_tem)
            x = torch.cat(x_list, 1)
            x = self.l1(x)
            return self.selu(self.l3(x)).reshape(-1, self.args['max_seq_len'], self.config.func(self.k, self.args)[0])


class datasetEmbedding(nn.Module):
    def __init__(self, config, k, args):
        super(datasetEmbedding, self).__init__()
        self.data_embeddings = EMNet(config, k, args)
        self.LayerNorm = BertLayerNorm(config.func(k, args)[0], eps=1e-12)

    def forward(self, dataset):
        embedding = self.data_embeddings(dataset)
        embedding = self.LayerNorm(embedding)
        return embedding


class adj_embedding(nn.Module):
    def __init__(self, config, k, args):
        super(adj_embedding, self).__init__()
        self.gc = graph_constructor(config.func(k, args)[0], config.func(k, args)[2], alpha=3, device='cuda:0',
                                    static_feat=None) if \
            args['cuda'] else graph_constructor(config.func(k, args)[0], config.func(k, args)[2], alpha=3, device='cpu',
                                                static_feat=None)
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.LayerNorm = BertLayerNorm(config.func(k, args)[2], eps=1e-12)
        self.layers = args['layers']
        self.args = args
        self.start_conv_1 = nn.Conv2d(in_channels=1,
                                      out_channels=args['conv_channels'],
                                      kernel_size=(1, 1),
                                      bias=True)
        for i in range(self.layers):
            self.gconv1.append(
                mixprop(args['conv_channels'], args['residual_channels'], args['gcn_depth'], args['dropout'],
                        args['propalpha']))
            self.gconv2.append(
                mixprop(args['conv_channels'], args['residual_channels'], args['gcn_depth'], args['dropout'],
                        args['propalpha']))
        self.end_conv_1 = nn.Conv2d(in_channels=args['residual_channels'],
                                    out_channels=args['end_channels'],
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=args['end_channels'],
                                    out_channels=config.func(k, args)[2],
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, x):
        adp = self.gc()
        x = self.start_conv_1(x)
        for i in range(self.layers):
            x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
        x = self.LayerNorm(x)
        x = F.selu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = self.LayerNorm(x)
        x = torch.einsum('ncvl->nlv', [x])
        return x


class TFTEmbeddings(nn.Module):
    """LayerNorm层, 见Transformer(一), 讲编码器(encoder)的第1部分"""
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, k, args, nnn_i):
        super(TFTEmbeddings, self).__init__()  # categray_counts每个类别的类别数
        WE = []
        self.args = args
        epsilon = 1e-8
        for i in range(args['categray_num']):  # categray_num共有多少类别
            WE.append(nn.Embedding(config.func_C(args, nnn_i)[i], config.func(k, args)[0]))
            # embedding矩阵初始化
            nn.init.orthogonal_(WE[i].weight)
            # embedding矩阵进行归一化
            WE[i].weight.data = \
                WE[i].weight.data.div(torch.norm(WE[i].weight, p=2, dim=1, keepdim=True).data + epsilon)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.func(k, args)[0], eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.word_embeddings = WE

    def forward(self, input_ids):
        """
        :param input_ids: 维度 [batch_size, sequence_length],!!!categray_Input
        :param positional_enc: 位置编码 [sequence_length, embedding_dimension]
        :param token_type_ids: BERT训练的时候, 第一句是0, 第二句是1
        :return: 维度 [batch_size, sequence_length, embedding_dimension]
        """
        # 字向量查表
        words_embeddings = []

        for i in range(input_ids.shape[-1]):
            # print(input_ids[:,:,i])
            words_embeddings.append(self.word_embeddings[i](input_ids[:, :, i].detach().cpu()))
            if self.args['cuda']:
                if i == 0:
                    embeddings = words_embeddings[i].cuda()
                else:
                    embeddings = embeddings + words_embeddings[i].cuda()
            else:
                if i == 0:
                    embeddings = words_embeddings[i]
                else:
                    embeddings = embeddings + words_embeddings[i]
        # embeddings: [batch_size, sequence_length, embedding_dimension]
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class EmbedingTotal(nn.Module):
    def __init__(self, config, k, args, nnn_i):
        super(EmbedingTotal, self).__init__()
        self.args = args
        if args['tec_Category']:
            self.embeddings = TFTEmbeddings(config, k, args, nnn_i)
        else:
            pass
        self.dense = nn.Linear(config.func(k, args)[0], -self.args['data_gap'])
        self.gc = adj_embedding(config, k, args)
        self.time_embedding = TemporalEmbedding(config.func(k, args)[0])
        self.datasetembedding = datasetEmbedding(config, k, args)
        self.informer = Informer(k - 3, k - 3, k - 3, args['max_seq_len'] - args['n_for'],
                                 args['max_seq_len'] - args['n_for'], args['max_seq_len'],
                                 factor=2, d_model=32, n_heads=4, e_layers=3, d_layers=2, d_ff=32,
                                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                                 output_attention=False, distil=True, mix=True,
                                 device="CPU")

    def forward(self, input_ids, time_cate, dataset, positional_enc):
        # informer 的组件
        if self.args['informer_switch']:
            datapred = self.informer(dataset[:, :(-self.args['n_for']), :], dataset[:, (self.args['n_for']):, :])
            dataset = torch.cat([dataset[:, (self.args['n_for']):, :], datapred], dim=1)
            # dataset = datapred
        # embedding层，必须要有
        dataset = self.datasetembedding(dataset)
        # tec分析+图神经网络
        if self.args['tec_Category']:
            embedding_output = self.embeddings(input_ids)
            if self.args['adj']:
                adj_embeddings = self.gc(dataset.permute(0, 2, 1).unsqueeze(1))
                adj_embeddings = nn.Softmax(dim=0)(adj_embeddings)
                dataset = dataset * adj_embeddings
            else:
                dataset = dataset
            embedding_output = dataset * embedding_output
        else:
            if self.args['adj']:
                adj_embeddings = self.gc(dataset.permute(0, 2, 1).unsqueeze(1))
                adj_embeddings = nn.Softmax(dim=0)(adj_embeddings)
                dataset = dataset + adj_embeddings
            else:
                dataset = dataset
            embedding_output = dataset
        # 时间种类编码
        if self.args['time_Category']:
            embedding_output2 = self.time_embedding(time_cate)
        else:
            embedding_output2 = torch.ones_like(embedding_output)
        # 综合三种输出
        embedding_output = positional_enc + embedding_output * embedding_output2
        return embedding_output
