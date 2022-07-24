# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.transformer
import thumt.models.deep_lstm
import thumt.models.bert
import thumt.models.emo_bert




def get_model(name):
    name = name.lower()

    models_dict = {
        "transformers": thumt.models.transformer.Transformer,
        "deep_lstm": thumt.models.deep_lstm.DeepLstm,
        "bert": thumt.models.bert.BERTModel,
        "emo_bert": thumt.models.emo_bert.EmoBertModel,
    }

    if name in models_dict.keys():
        return models_dict[name]

    # model = models_dict. [name]
    #
    # if name == "transformer":
    #     return thumt.models.transformer.Transformer
    # elif name == "deep_lstm":
    #     return thumt.models.deep_lstm.DeepLstm
    # elif name == "bert":
    #     return thumt.models.bert.BERTModel
    # elif name == "emo_bert":
    #     return thumt.models.emo_bert.EmoBertModel
    # else:
    raise LookupError("Unknown model %s" % name)
