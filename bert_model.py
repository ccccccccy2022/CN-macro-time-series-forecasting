# -*- coding = utf-8 -*-
# @time:2021/11/26 16:23
# Author:Tjd_T
# @File:bert_model.py
# @Software:PyCharm

from model_tools.loss.dilate_loss import dilate_loss
from model_tools.bert.bert_component import BertPreTrainedModel, BertLayerNorm, BertPooler, BertEncoder,BertDecoder
from model_tools.bert.bert_utiles import EmbedingTotal
import torch.nn.functional as F
import torch
from torch import nn


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        config: a BertConfig class instance with the configuration to build a new model
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
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, k, args, nnn_i, pn):
        super(BertModel, self).__init__(config, k, args)
        self.args = args
        self.pn = pn
        self.encoder = BertEncoder(config, k, args)
        self.pooler = BertPooler(config, k, args)
        self.apply(self.init_bert_weights)

        self.LayerNorm = BertLayerNorm(config.func(k, args)[0], eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.reembed = EmbedingTotal(config, k, args, nnn_i)
        #TODO 20220211 增加decoder
        self.decoder=BertDecoder(config, k, args)
        #TODO 20220715 similar0525
        self.dense2 = nn.Linear(self.args['max_seq_len'],-self.args['data_gap'])
        self.dense = nn.Linear(config.func(k, args)[0], -self.args['data_gap'])

    def forward(self, input_ids, time_cate, dataset, positional_enc, labels, attention_mask=None,
                output_all_encoded_layers=True, get_attention_matrices=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(dataset[:, :, 0])

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # 给注意力矩阵里padding的无效区域加一个很大的负数的偏置, 为了使softmax之后这些无效区域仍然为0, 不参与后续计算
        # 自己统合的类
        embedding_output = self.reembed(input_ids, time_cate, dataset, positional_enc)
        embedding_output = self.LayerNorm(embedding_output)
        embedding_output = self.dropout(embedding_output)

        # 经过所有定义的transformer block之后的输出
        encoded_layers, all_attention_matrices = self.encoder(embedding_output, extended_attention_mask,
                                                              output_all_encoded_layers=output_all_encoded_layers,
                                                              get_attention_matrices=get_attention_matrices)
        # [-1]为最后一个transformer block的隐藏层的计算结果
        # sequence_output = encoded_layers[-1]
        #TODO 20220211 增加了decoder labels_sr是重构label，self.args['data_gap']目前设的都是-1，就是一次预测一个点；label本来是y1到yt，这里的label_sr是往前移动一个点，第一个点用y1重复填充保证维度不变：y1,y1到yt-1
        #增加一个数据label_s12,放在args里，如果self.args['label_s12']==None,pass;else,labels=self.args['label_s12'],data_creation里增加
        #TODO 20220511
        labels_sr = torch.cat([labels[:,0:-self.args['data_gap']],labels[:,:self.args['data_gap']]],dim=1).to(torch.float32)
        Labels_sr =self.reembed(input_ids, time_cate, torch.cat([labels_sr.unsqueeze(2) for x in range(dataset.shape[2])],dim=2), positional_enc)
        sequence_output = self.decoder(encoded_layers, Labels_sr,attention_mask, get_attention_matrices=False)
        sequence_output2 = sequence_output[-1]
        # TODO 20220511
        # TODO 20220511,20220525
        try:
            first_token_tensor = self.dense2(sequence_output2.permute(2,1,0)).permute(2,1,0)
        except:
            first_token_tensor=self.dense2(sequence_output2.permute(0,2,1)).permute(0,2,1)
        # first_token_tensor = sequence_output2[:,-1]
        predictions = self.dense(first_token_tensor).squeeze(0)

        return predictions


