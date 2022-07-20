# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 14:51
# @Author  : YuJie
# @Email   : 162095214@qq.com
# @File    : deep_lstm.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import thumt.utils as utils
import thumt.modules as modules


class AttentionSubLayer(modules.Module):

    def __init__(self, params, hidden_size, name="attention"):
        super(AttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.attention = modules.MultiHeadAttention(
                hidden_size, params.num_heads, params.attention_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x, bias, memory=None, state=None):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.training or state is None:
            y = self.attention(y, bias, memory, None)
        else:
            kv = [state["k"], state["v"]]
            y, k, v = self.attention(y, bias, memory, kv)
            state["k"], state["v"] = k, v

        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return y
        else:
            return self.layer_norm(y)


class DHEASubLayer(modules.Module):
    def __init__(self, params, name="dhea_layer"):
        super(DHEASubLayer, self).__init__(name=name)

        with utils.scope(name):
            self.c_attention = AttentionSubLayer(params, params.hidden_size)
            self.z_attention = AttentionSubLayer(params, params.hidden_size)
            self.hybird_attention = AttentionSubLayer(params, params.hidden_size * 2)
            self.lstm = LSTMSubLayer(params)

    def forward(self, x, enc_bias, dec_bias, memory, state, way="sum"):
        if way == "hybird":
            x = self.hybird_attention(torch.cat((memory, x)), dec_bias, state=state)
        else:
            z = self.z_attention(x, dec_bias, state=state)
            c = self.c_attention(x, enc_bias, memory)
            if way == "sum":
                x = z + c
            elif way == "gate":
                weight = torch.sigmoid(torch.cat((c, z)))
                x = weight * c + (1 - weight) * z
            else:
                raise TypeError("Not correct way to combine z and c in DHEA")
        x = self.lstm(x)
        return x


class LSTMSubLayer(modules.Module):
    def __init__(self, params, forward_dir=True, name="lstm_layer"):
        super(LSTMSubLayer, self).__init__(name=name)
        self.input_size = params.hidden_size
        self.hidden_size = params.hidden_size
        self.forward_dir = forward_dir

        with utils.scope(name):
            self.lstm = modules.LSTMCell(self.input_size, self.hidden_size)

    def flip(self, x):
        """
        Args:
            x: [batch, length, hidden_size]
            state:
        Returns:
        """
        pad = x[0][-1]
        output = []
        for sentence in x:
            end = sentence.shape[0] - 1
            for i in range(sentence.shape[0] - 1, -1, -1):
                if not sentence[i].equal(pad):
                    end = i
                    break
            content = sentence[:end + 1]
            padding = sentence[end + 1:]
            content = torch.flip(content, dims=[0])
            content = torch.cat((content, padding), dim=0)
            output.append(torch.unsqueeze(content, dim=0))
        output=torch.cat(output,dim=0)
        return output

    def forward(self, x, state=None):
        """
        Args:
            x: [batch, length, hidden_size]
            state:
        Returns:
        """
        if not self.forward_dir:
            x = self.flip(x)

        # [batch, length, input_size] -> [length, batch, input_size]
        x = torch.transpose(x, 0, 1)

        batch_size = x.shape[1]
        if state is None:
            state = self.lstm.init_state(batch_size, dtype=torch.float, device=torch.device("cuda"))

        output = []
        for seq in range(x.shape[0]):
            hidden, state = self.lstm(x[seq], state)
            output.append(torch.unsqueeze(hidden, dim=0))

        output = torch.cat(output, dim=0)

        if not self.forward_dir:
            output = self.flip(output)

        return output


class FFNSubLayer(modules.Module):

    def __init__(self, params, name="ffn_layer"):
        super(FFNSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.ffn_layer = modules.FeedForward(params.hidden_size,
                                                 params.filter_size,
                                                 dropout=params.relu_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        y = self.ffn_layer(y)
        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class DeepLstmEncoderLayer(modules.Module):
    def __init__(self, params, forward_dir=True, name="layer"):
        super(DeepLstmEncoderLayer, self).__init__(name=name)
        self.forward_dir = forward_dir

        with utils.scope(name):
            self.ffn = FFNSubLayer(params)
            self.lstm = LSTMSubLayer(params, self.forward_dir)

    def forward(self, x):
        """
        Args:
            x: [batch, length, input_size]
        Returns:
        """
        x = self.lstm(x)
        x = self.ffn(x)
        return x


class DeepLstmDecoderLayer(modules.Module):
    def __init__(self, params, name="layer"):
        super(DeepLstmDecoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.ffn = FFNSubLayer(params)
            self.dhea = DHEASubLayer(params)

    def forward(self, x, dec_bias, enc_bias, memory, state):
        x = self.ffn(x)
        x = self.dhea(x, dec_bias, enc_bias, memory, state)
        return x


class DeepLstmEncoder(modules.Module):
    def __init__(self, params, name="encoder"):
        super(DeepLstmEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                DeepLstmEncoderLayer(params, name="layer_%d" % i, forward_dir=True if i % 2 == 0 else False)
                for i in range(params.num_encoder_layers)])
            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x,bias):
        """
        Args:
            x: [batch, length, input_size]
        Returns:
        """
        for layer in self.layers:
            x = layer(x)
        if self.normalization == "before":
            x = self.layer_norm(x)
        return x


class DeepLstmDecoder(modules.Module):
    def __init__(self, params, name="decoder"):
        super(DeepLstmDecoder, self).__init__(name=name)
        self.normalization = params.normalization
        with utils.scope(name):
            self.layers = nn.ModuleList([
                DeepLstmDecoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_decoder_layers)])
            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x, attn_bias, encdec_bias, memory, state=None):
        for i, layer in enumerate(self.layers):
            if state is not None:
                x = layer(x, attn_bias, encdec_bias, memory,
                          state["decoder"]["layer_%d" % i])
            else:
                x = layer(x, attn_bias, encdec_bias, memory, None)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class DeepLstm(modules.Module):
    def __init__(self, params, name="deep_lstm"):
        super(DeepLstm, self).__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = DeepLstmEncoder(params)
            self.decoder = DeepLstmDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
        self.reset_parameters()

    def build_embedding(self, params):
        svoc_size = len(params.vocabulary["source"])
        tvoc_size = len(params.vocabulary["target"])

        if params.shared_source_target_embedding and svoc_size != tvoc_size:
            raise ValueError("Cannot share source and target embedding.")

        if not params.shared_embedding_and_softmax_weights:
            self.softmax_weights = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.softmax_weights, "softmax_weights")

        if not params.shared_source_target_embedding:
            self.source_embedding = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.target_embedding = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.source_embedding, "source_embedding")
            self.add_name(self.target_embedding, "target_embedding")
        else:
            self.weights = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.add_name(self.weights, "weights")

        self.bias = torch.nn.Parameter(torch.zeros([params.hidden_size]))
        self.add_name(self.bias, "bias")

    @property
    def src_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.source_embedding

    @property
    def tgt_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.target_embedding

    @property
    def softmax_embedding(self):
        if not self.params.shared_embedding_and_softmax_weights:
            return self.softmax_weights
        else:
            return self.tgt_embedding

    def reset_parameters(self):
        nn.init.normal_(self.src_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.normal_(self.tgt_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)

        if not self.params.shared_embedding_and_softmax_weights:
            nn.init.normal_(self.softmax_weights, mean=0.0,
                            std=self.params.hidden_size ** -0.5)

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias
        inputs = nn.functional.dropout(self.encoding(inputs), self.dropout,
                                       self.training)

        enc_attn_bias = enc_attn_bias.to(inputs)

        encoder_output = self.encoder(inputs, enc_attn_bias)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])

        targets = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)

        decoder_input = torch.cat(
            [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
             targets[:, 1:, :]], dim=1)
        decoder_input = nn.functional.dropout(self.encoding(decoder_input),
                                              self.dropout, self.training)

        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(targets)

        if mode == "infer":
            decoder_input = decoder_input[:, -1:, :]
            dec_attn_bias = dec_attn_bias[:, :, -1:, :]

        decoder_output = self.decoder(decoder_input, dec_attn_bias,
                                      enc_attn_bias, encoder_output, state)

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1)

        return logits, state

    def forward(self, features, labels, mode="train", level="sentence"):
        mask = features["target_mask"]

        state = self.empty_state(features["target"].shape[0],
                                 labels.device)
        state = self.encode(features, state)
        logits, _ = self.decode(features, state, mode=mode)
        loss = self.criterion(logits, labels)
        mask = mask.to(torch.float32)

        # Prevent FP16 overflow
        if loss.dtype == torch.float16:
            loss = loss.to(torch.float32)

        if mode == "eval":
            if level == "sentence":
                return -torch.sum(loss * mask, 1)
            else:
                return torch.exp(-loss) * mask - (1 - mask)

        return (torch.sum(loss * mask) / torch.sum(mask)).to(logits)

    def empty_state(self, batch_size, device):
        state = {
            "decoder": {
                "layer_%d" % i: {
                    "k": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device),
                    "v": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device)
                } for i in range(self.num_decoder_layers)
            }
        }

        return state

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        # 返回一个上三角矩阵，diagonal=1
        # 如果diagonal为空，输⼊矩阵保留主对⾓线与主对⾓线以上的元素；
        # 如果diagonal为正数n，输⼊矩阵保留主对⾓线以上除去n⾏的元素；
        # 如果diagonal为负数-n，输⼊矩阵保留主对⾓线与主对⾓线以上与主对⾓线下⽅h⾏对⾓线的元素
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

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
    def base_params_v2():
        params = DeepLstm.base_params()
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.learning_rate = 12e-4
        params.warmup_steps = 8000
        params.normalization = "before"
        params.adam_beta2 = 0.997

        return params

    @staticmethod
    def big_params():
        params = DeepLstm.base_params()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 5e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def big_params_v2():
        params = DeepLstm.base_params_v2()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 7e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def default_params(name=None):
        if name == "base":
            return DeepLstm.base_params()
        elif name == "base_v2":
            return DeepLstm.base_params_v2()
        elif name == "big":
            return DeepLstm.big_params()
        elif name == "big_v2":
            return DeepLstm.big_params_v2()
        else:
            return DeepLstm.base_params()
