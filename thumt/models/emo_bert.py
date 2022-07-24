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
from thumt.models.bert import BERTEncoder


class EmoBertModel(modules.Module):

    def __init__(self, params, name="emo_bert"):
        super(EmoBertModel, self).__init__(name=name)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)

        with utils.scope(name):
            self.encoder = BERTEncoder(params)

            self.emo_hidden = nn.Sequential(nn.Linear(params.hidden_size, params.hidden_size),
                                            nn.Tanh())

            self.output = nn.Linear(params.hidden_size, 3)

    def forward(self, tokens, segments, bias, labels):
        x = self.encoder(tokens, segments, bias)

        x = self.output(self.emo_hidden(x[:, 0, :]))

        loss = self.criterion(x, labels)
        precision, _, _ = self.cal_precision(x, labels)
        loss = loss.sum(0)

        return loss, precision

    @staticmethod
    def cal_precision(x, labels):
        hat = torch.nn.functional.softmax(x, dim=1)

        res = []
        for example in hat:
            if example[0] > example[1]:
                res.append(0)
            else:
                res.append(1)
        res = torch.tensor(res)
        ac = 0
        total = labels.shape[0]
        for i in range(labels.shape[0]):
            if labels[i] == res[i]:
                ac += 1
        precision = ac * 1.0 / total
        return precision, ac, total

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
        return EmoBertModel.base_params()
