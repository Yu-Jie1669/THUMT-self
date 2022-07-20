# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.transformer
import thumt.models.deep_lstm
import thumt.models.bert


def get_model(name):
    name = name.lower()

    if name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "deep_lstm":
        return thumt.models.deep_lstm.DeepLstm
    elif name == "bert":
        return thumt.models.bert.BERTModel
    else:
        raise LookupError("Unknown model %s" % name)
