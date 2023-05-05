# -*- coding = utf-8 -*-
# @time:2021/11/26 16:18
# Author:Tjd_T
# @File:bert_component.py
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


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "selu": torch.nn.functional.selu}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 # vocab_size, # 字典字数
                 # hidden_size=k, # 隐藏层维度也就是字向量维度
                 # max_seq_len =args['max_seq_len'],
                 num_hidden_layers=3,  # transformer block 的个数
                 # num_attention_heads=args['head'], # 注意力机制"头"的个数
                 # intermediate_size=k*2, # feedforward层线性映射的维度
                 hidden_act="selu",  # 激活函数
                 hidden_dropout_prob=0.0,  # dropout的概率
                 attention_probs_dropout_prob=0.0,
                 max_position_embeddings=512 * 2,
                 # state_dict_dir = os.getcwd(),
                 initializer_range=0.02):
        # type_vocab_size=256, # 用来做next sentence预测,
        # 这里预留了256个分类, 其实我们目前用到的只有0和1
        # 用来初始化模型参数的标准差

        # self.vocab_size = vocab_size
        # self.max_seq_len = max_seq_len
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        # self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        # self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        # self.state_dict_dir = state_dict_dir

    def func(self, k, args):
        #TODO 20220511
        self.hidden_size = (k + args['categray_num'])
        self.num_attention_heads = args['head']
        self.max_seq_len = args['max_seq_len']
        self.intermediate_size = (k + args['categray_num']) * 2
        return self.hidden_size, self.num_attention_heads, self.max_seq_len, self.intermediate_size

    def func_C(self, args, i):
        self.categray_counts = args['categray_counts_list'][i]  # 是个list，每个list里对应分类变量数量
        return self.categray_counts


class BertEmbeddings(nn.Module):
    """LayerNorm层, 见Transformer(一), 讲编码器(encoder)的第1部分"""
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, k, args):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.func(k, args)[0], padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.func(k, args)[0])
        # embedding矩阵初始化
        nn.init.orthogonal_(self.word_embeddings.weight)
        nn.init.orthogonal_(self.token_type_embeddings.weight)

        # embedding矩阵进行归一化
        epsilon = 1e-8
        self.word_embeddings.weight.data = \
            self.word_embeddings.weight.data.div(
                torch.norm(self.word_embeddings.weight, p=2, dim=1, keepdim=True).data + epsilon)
        self.token_type_embeddings.weight.data = \
            self.token_type_embeddings.weight.data.div(
                torch.norm(self.token_type_embeddings.weight, p=2, dim=1, keepdim=True).data + epsilon)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.func(k, args)[0], eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, positional_enc, token_type_ids=None):
        """
        :param input_ids: 维度 [batch_size, sequence_length]
        :param positional_enc: 位置编码 [sequence_length, embedding_dimension]
        :param token_type_ids: BERT训练的时候, 第一句是0, 第二句是1
        :return: 维度 [batch_size, sequence_length, embedding_dimension]
        """
        # 字向量查表
        words_embeddings = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + positional_enc + token_type_embeddings
        # embeddings: [batch_size, sequence_length, embedding_dimension]
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """自注意力机制层, 见Transformer(一), 讲编码器(encoder)的第2部分"""

    def __init__(self, config, k, args):
        super(BertSelfAttention, self).__init__()
        # 判断embedding dimension是否可以被num_attention_heads整除
        if config.func(k, args)[0] % config.func(k, args)[1] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.func(k, args)[0], config.func(k, args)[1]))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.func(k, args)[0] / config.func(k, args)[1])
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Q, K, V线性映射
        self.query = nn.Linear(config.func(k, args)[0], self.all_head_size)
        self.key = nn.Linear(config.func(k, args)[0], self.all_head_size)
        self.value = nn.Linear(config.func(k, args)[0], self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.args = args

    def transpose_for_scores(self, x):
        # 输入x为QKV中的一个, 维度: [batch_size, seq_length, embedding_dim]
        # 输出的维度经过reshape和转置: [batch_size, num_heads, seq_length, embedding_dim / num_heads]
        # print(x.shape)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # print(new_x_shape)
        # print(x.shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, get_attention_matrices=False):  #
        # Q, K, V线性映射
        # Q, K, V的维度为[batch_size, seq_length, num_heads * embedding_dim]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # 把QKV分割成num_heads份
        # 把维度转换为[batch_size, num_heads, seq_length, embedding_dim / num_heads]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # Q与K求点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores: [batch_size, num_heads, seq_length, seq_length]
        # 除以K的dimension, 开平方根以归一为标准正态分布
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if self.args['cuda']:
            mask = torch.tril(torch.ones(attention_scores.shape[2], attention_scores.shape[3])).cuda()
        else:
            mask = torch.tril(torch.ones(attention_scores.shape[2], attention_scores.shape[3]))
        attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # print('attention_mask')
        # print(attention_mask.shape)

        try:
            attention_scores = attention_scores
        except:
            attention_scores = attention_scores
        # attention_mask 注意力矩阵mask: [batch_size, 1, 1, seq_length]
        # 元素相加后, 会广播到维度: [batch_size, num_heads, seq_length, seq_length]

        # softmax归一化, 得到注意力矩阵
        # Normalize the attention scores to probabilities.
        attention_probs_ = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs_)

        # 用注意力矩阵加权V
        context_layer = torch.matmul(attention_probs, value_layer)
        # 把加权后的V reshape, 得到[batch_size, length, embedding_dimension]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 输出attention矩阵用来可视化
        if get_attention_matrices:
            return context_layer, attention_probs_
        return context_layer, None


class BertLayerNorm(nn.Module):
    """LayerNorm层, 见Transformer(一), 讲编码器(encoder)的第3部分"""

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfOutput(nn.Module):
    # 封装的LayerNorm和残差连接, 用于处理SelfAttention的输出
    def __init__(self, config, k, args):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.func(k, args)[0], config.func(k, args)[0])
        self.LayerNorm = BertLayerNorm(config.func(k, args)[0], eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    # 封装的多头注意力机制部分, 包括LayerNorm和残差连接
    def __init__(self, config, k, args):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config, k, args)
        self.output = BertSelfOutput(config, k, args)

    def forward(self, input_tensor, attention_mask, get_attention_matrices=False):  #
        self_output, attention_matrices = self.self(input_tensor, attention_mask,
                                                    get_attention_matrices=get_attention_matrices)  #
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_matrices


class BertIntermediate(nn.Module):
    # 封装的FeedForward层和激活层
    def __init__(self, config, k, args):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.func(k, args)[0], config.func(k, args)[3])
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    # 封装的LayerNorm和残差连接, 用于处理FeedForward层的输出
    def __init__(self, config, k, args):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.func(k, args)[3], config.func(k, args)[0])
        self.LayerNorm = BertLayerNorm(config.func(k, args)[0], eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    # 一个transformer block
    def __init__(self, config, k, args):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config, k, args)
        self.intermediate = BertIntermediate(config, k, args)
        self.output = BertOutput(config, k, args)

    def forward(self, hidden_states, attention_mask, get_attention_matrices=False):  #
        # Attention层(包括LayerNorm和残差连接)
        attention_output, attention_matrices = self.attention(hidden_states, attention_mask,
                                                              get_attention_matrices=get_attention_matrices)  #
        # FeedForward层
        intermediate_output = self.intermediate(attention_output)
        # LayerNorm与残差连接输出层
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_matrices
#TODO 20220111
class SublayerConnection(nn.Module):
    def __init__(self, config, k, args):
        super(SublayerConnection, self).__init__()
        self.LayerNorm = BertLayerNorm(config.func(k, args)[0], eps=1e-12)
    def forward(self,x,sublayers):
        return x+sublayers
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class DecoderLayer(nn.Module):
    # 一个transformer block
    def __init__(self, config, k, args):
        super(DecoderLayer, self).__init__()
        self.attention = BertAttention(config, k, args)
        self.intermediate = BertIntermediate(config, k, args)
        self.output = BertOutput(config, k, args)
        self.sublayer = clones(SublayerConnection(config, k, args), 3)
        self.LayerNorm = BertLayerNorm(config.func(k, args)[0], eps=1e-12)

    def forward(self, layer_output, labels_sr,attention_mask, get_attention_matrices=False):  #
        # Attention层(包括LayerNorm和残差连接)
        self.self_att =layer_output+labels_sr
        self.linear = nn.Linear(self.self_att.shape[2],self.self_att.shape[2])
        layer_output=self.sublayer[0](self.self_att,self.linear(self.LayerNorm(self.self_att)))
        layer_output= self.sublayer[1](layer_output,self.attention(self.LayerNorm(layer_output), attention_mask,
                                                              get_attention_matrices=get_attention_matrices)[0])  #
        # FeedForward层
        intermediate_output =self.intermediate(self.LayerNorm(layer_output))
        # LayerNorm与残差连接输出层
        layer_output2 = self.output(intermediate_output, layer_output)
        return layer_output2
class BertDecoder(nn.Module):
    def __init__(self, config, k, args):
        super(BertDecoder, self).__init__()
        layer =DecoderLayer(config, k, args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
    def forward(self,layer_output, labels_sr,attention_mask, get_attention_matrices=False):
        all_Decoder_layers = []
        for i,layer_module in enumerate(self.layer):
            layer_output2 = layer_module(layer_output[i], labels_sr,attention_mask, get_attention_matrices=False)
            all_Decoder_layers.append(layer_output2)
        return all_Decoder_layers

class BertEncoder(nn.Module):
    # transformer blocks * N
    def __init__(self, config, k, args):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config, k, args)
        # 复制N个transformer block
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, get_attention_matrices=False):
        """#
        :param output_all_encoded_layers: 是否输出每一个transformer block的隐藏层计算结果
        :param get_attention_matrices: 是否输出注意力矩阵, 可用于可视化
        """
        all_attention_matrices = []
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_matrices = layer_module(hidden_states, attention_mask,
                                                             get_attention_matrices=get_attention_matrices)  # , attention_mask
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_attention_matrices.append(attention_matrices)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_attention_matrices.append(attention_matrices)
        return all_encoder_layers, all_attention_matrices


class BertPooler(nn.Module):
    """Pooler是把隐藏层(hidden state)中对应#CLS#的token的一条提取出来的功能"""

    def __init__(self, config, k, args):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.func(k, args)[0], config.func(k, args)[0])
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 线性映射, 激活, LayerNorm
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        # 线性映射, 激活, LayerNorm
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        """上面是创建一个线性映射层, 把transformer block输出的[batch_size, seq_len, embed_dim]
        映射为[batch_size, seq_len, vocab_size], 也就是把最后一个维度映射成字典中字的数量, 
        获取MaskedLM的预测结果, 注意这里其实也可以直接矩阵成embedding矩阵的转置, 
        但一般情况下我们要随机初始化新的一层参数
        """
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


# BERT的训练中通过隐藏层输出Masked LM的预测和Next Sentence的预测
class BertPreTrainingHeads(nn.Module):
    """
    BERT的训练中通过隐藏层输出Masked LM的预测和Next Sentence的预测
    """

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()

        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        # 把transformer block输出的[batch_size, seq_len, embed_dim]
        # 映射为[batch_size, seq_len, vocab_size]
        # 用来进行MaskedLM的预测
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        # 用来把pooled_output也就是对应#CLS#的那一条向量映射为2分类
        # 用来进行Next Sentence的预测

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# 用来初始化模型参数
class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
        用来初始化模型参数
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear)):
            # 初始线性映射层的参数为正态分布
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            torch.nn.init.xavier_uniform_(module.weight.data, gain=1)
        elif isinstance(module, BertLayerNorm):
            # 初始化LayerNorm中的alpha为全1, beta为全0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 初始化偏置为0
            module.bias.data.zero_()




class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.
    Params:
        config: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)
        self.vocab_size = config.vocab_size
        self.next_loss_func = CrossEntropyLoss()
        self.mlm_loss_func = CrossEntropyLoss(ignore_index=0)

    def compute_loss(self, predictions, labels, num_class=2, ignore_index=-100):
        loss_func = CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(predictions.view(-1, num_class), labels.view(-1))

    def forward(self, input_ids, positional_enc, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, positional_enc, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        mlm_preds, next_sen_preds = self.cls(sequence_output, pooled_output)
        return mlm_preds, next_sen_preds
