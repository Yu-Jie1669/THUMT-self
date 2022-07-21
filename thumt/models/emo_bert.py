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
from thumt.models.bert import BERTEncoder


class EmoBertModel(modules.Module):

    def __init__(self, params, name="emo_bert"):
        super(EmoBertModel, self).__init__(name=name)

        with utils.scope(name):
            self.bert = BERTEncoder(params)
            self.output = nn.Linear(params.hidden_size, 3)

    def forward(self, tokens, segments, bias):
        x = self.bert(tokens, segments, bias)
        x = self.output(x)
        return x
