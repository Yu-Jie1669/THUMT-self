from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

import torch
import torch.nn as nn
from torch import sigmoid

import thumt.utils as utils
import thumt.modules as modules
from thumt.models.transformer import TransformerEncoderLayer as EncoderLayer
from thumt.modules.embedding import PositionalEmbedding


class BERTEncoder(modules.Module):

    def __init__(self, params, vocab_size=21128, name='bert_encoder'):
        super(BERTEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            # nn.Embedding
            # output_size = input_size * hidden_size(embedding_dim)
            self.token_embedding = nn.Embedding(vocab_size, params.hidden_size)
            self.segment_embedding = nn.Embedding(2, params.hidden_size)

            self.pos_embedding = PositionalEmbedding()

            self.layers = nn.ModuleList(
                [EncoderLayer(params, name='layer_%id' % i)
                 for i in range(params.num_encoder_layers)]
            )

            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, tokens, segments, bias):
        # 输入是 词嵌入+段嵌入+位置嵌入
        x = self.token_embedding(tokens)

        if not segments is None:
            x += self.segment_embedding(segments)

        x = self.pos_embedding(x)

        bias = torch.reshape(bias, (bias.shape[0], 1, 1, bias.shape[-1]))

        for layer in self.layers:
            x = layer(x, bias)

        if self.normalization == "before":
            x = self.layer_norm(x)
        return x


class MaskLM(modules.Module):

    def __init__(self, params, name='mask_lm'):
        super(MaskLM, self).__init__(name=name)

        self.vacob_size = 21128

        with utils.scope(name):
            # 参数：input_size,hidden_size,output_size（词表大小）,dropout*
            self.mlp = modules.FeedForward(params.hidden_size, params.hidden_size, self.vacob_size)

    def forward(self, x, pred_positions):
        """
        Args:
            x: input
            pred_positions: [batch_size,num_pred]

        Returns:
        """
        num_pred = pred_positions.shape[1]
        # [batch_size,num_pred] -> [batch_size*num_pred]
        pred_positions = pred_positions.reshape(-1)

        batch_size = x.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # ->[0,0,0...1,1,1...,2,2,2...](num_pred*batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred)

        # -> [batch0pos0,batch0pos1,batch0pos2,...batch1pos0...]
        masked_input = x[batch_idx, pred_positions]
        # ->[batch_size,num_pred,1]
        masked_input = masked_input.reshape((batch_size, num_pred, -1))

        mlm = self.mlp(masked_input)
        return mlm


class NextSentencePred(modules.Module):
    def __init__(self, params, name="nsp"):
        super(NextSentencePred, self).__init__(name=name)

        with utils.scope(name):
            self.output = nn.Linear(params.hidden_size, 2)

    def forward(self, x):
        # [batch_size,hidden_size]
        return self.output(x)


class BERTModel(modules.Module):
    def __init__(self, params, name="bert"):
        super(BERTModel, self).__init__(name=name)

        self.vocab_size = 21128

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)

        with utils.scope(name):
            self.encoder = BERTEncoder(params)
            self.hidden = nn.Sequential(nn.Linear(params.hidden_size, params.hidden_size),
                                        nn.Tanh())

            self.mlm = MaskLM(params)
            self.nsp = NextSentencePred(params)

    def forward(self, tokens, segments, bias, mlm_weights, nsp_y, mlm_y,
                pred_positions=None):
        encoded_x = self.encoder(tokens, segments, bias)
        if pred_positions is not None:
            mlm_y_hat = self.mlm(encoded_x, pred_positions)
        else:
            mlm_y_hat = None

        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_y_hat = self.nsp(self.hidden(encoded_x[:, 0, :]))

        # 计算遮蔽语言模型损失
        mlm_l = self.criterion(mlm_y_hat.reshape(-1, self.vocab_size), mlm_y.reshape(-1)) * mlm_weights.reshape(-1, 1)
        mlm_l = mlm_l.sum() / (mlm_weights.sum() + 1e-8)
        # 计算下一句子预测任务的损失
        nsp_l = self.criterion(nsp_y_hat, nsp_y)
        nsp_l = nsp_l.mean(0)

        l = mlm_l + nsp_l

        mlm_ac, mlm_total, mlm_precision, nsp_ac, nsp_total, nsp_precision = self.cal_precision(mlm_y, mlm_y_hat, nsp_y,
                                                                                                nsp_y_hat)
        return mlm_l, nsp_l, l, mlm_precision, nsp_precision

    @staticmethod
    def cal_precision(mlm_y, mlm_y_hat, nsp_y, nsp_y_hat):
        nsp_y_hat = torch.nn.functional.softmax(nsp_y_hat, dim=1)
        mlm_y_hat = torch.nn.functional.softmax(mlm_y_hat, dim=2)

        nsp_res = []
        for example in nsp_y_hat:
            if example[0] > example[1]:
                nsp_res.append(0)
            else:
                nsp_res.append(1)
        nsp_res = torch.tensor(nsp_res)
        nsp_ac = 0
        nsp_total = nsp_y.shape[0]
        for i in range(nsp_y.shape[0]):
            if nsp_y[i] == nsp_res[i]:
                nsp_ac += 1
        nsp_precision = nsp_ac * 1.0 / nsp_total

        mlm_ac = 0
        mlm_total = 0
        mlm_res = []
        for example in mlm_y_hat:
            mlm_res.append(torch.max(example, 1)[1])
        # mlm_res = torch.tensor(mlm_res)
        for i in range(mlm_y.shape[0]):
            for j in range(len(mlm_y[i])):
                if mlm_y[i][j] == 0:
                    break
                if mlm_y[i][j] == mlm_res[i][j]:
                    mlm_ac += 1
                mlm_total += 1
        mlm_precision = mlm_ac * 1.0 / mlm_total

        return mlm_ac, mlm_total, mlm_precision, nsp_ac, nsp_total, nsp_precision

    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            normalization="after",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            warmup_steps=4000,
            train_steps=100000,
            learning_rate=7e-4,
            learning_rate_schedule="linear_warmup_rsqrt_decay",
            batch_size=4096,
            fixed_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0
        )

        return params

    @staticmethod
    def default_params(name=None):
        return BERTModel.base_params()


if __name__ == "__main__":
    #     x = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11], [11, 10, 9, 8, 7, 6, 5, 4]])
    #     pred_positions = torch.tensor([[1, 4, 6], [2, 3, 7]])
    #     num_pred = pred_positions.shape[1]
    #     # [batch_size,num_pred] -> [batch_size*num_pred]
    #     pred_positions = pred_positions.reshape(-1)
    #
    #     batch_size = x.shape[0]
    #     batch_idx = torch.arange(0, batch_size)
    #     # ->[0,0,0...1,1,1...,2,2,2...](num_pred*batch_size)
    #     batch_idx = torch.repeat_interleave(batch_idx, num_pred)
    #
    #     # -> [batch0pos0,batch0pos1,batch0pos2,...batch1pos0...]
    #     masked_input = x[batch_idx, pred_positions]
    #     # ->[batch_size,num_pred,1]
    #     masked_input = masked_input.reshape((batch_size, num_pred, -1))

    a = 0
