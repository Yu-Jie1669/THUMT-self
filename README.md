#### 文件目录

还没有写test的部分

    ├─bert-train
    ├─data
    │  ├─chinese 预训练数据 （电影评论） 词表
    │  ├─emo 情感分析数据
    ├─thumt
    │  ├─bin
    │  │  └─bert_pretrain.py BERT预训练
    │  │  └─emo_train.py 情感分析训练
    │  ├─data
    │  │  └─samples.py 电影Dataset（BERT的mask和nsp数据） 情感Dataset
    │  └─models
    │     └─bert.py BERT模型
    │     └─emo_bert.py 情感分类模型
    └─train

samples.py 数据集

```python

# coding=utf-8

import random
import re

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from thumt.data.vocab import Vocabulary


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """
       获得词元序列以及segments序列
    """
    tokens = tokens_a
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a))
    if tokens_b is not None:
        tokens += tokens_b[1:]
        segments += [1] * (len(tokens_b) - 1)
    return tokens, segments


class FilmDataset(Dataset):

    def __init__(self, input_path, params, vocab_path, max_len):
        with open(input_path, 'r', encoding='UTF-8') as f:
            paragraphs = f.readlines()
        # paragraphs ->[paragraph_count,sentences_count]
        paragraphs = [self.cut_sent(paragraph) for paragraph in paragraphs]

        self.vocab = params.vocabulary['source']
        
        # encode 出来前缀自带 [CLS] 后缀自带 [SEP]，这边调的是BERTTokenizer
        self.input_ids = self.encode(max_len, vocab_path, paragraphs)
        
        # 获取下一句子预测任务的数据
        examples = []

        # examples->[(tokens(id), segments, is_next) * len(example)]
        for paragraph_ids in self.input_ids:
            # get_nsp_data根据BERT论文中的做法 50%正例，50%负例
            examples.extend(self.get_nsp_data(
                paragraph_ids, self.input_ids, max_len))

        # 获得MASK任务数据
        # 也是根据论文中做法 0.15 再分0.8 0.1 0.1
        examples = [(self.get_mlm_data(tokens, self.vocab)
                     + (segments, is_next))
                    for tokens, segments, is_next in examples]

        # examples->[[mlm_input_token_ids, pred_positions, mlm_pred_labels,segments,is_next],
        #            [],...]
        # label:id

        # 填充到统一维度
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels, self.attention_mask) = self.pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return self.all_token_ids[idx], self.all_segments[idx], \
               self.valid_lens[idx], self.all_pred_positions[idx], \
               self.all_mlm_weights[idx], self.all_mlm_labels[idx], \
               self.nsp_labels[idx], self.attention_mask[idx]

    def __len__(self):
        return len(self.all_token_ids)

    def get_data(self):
        # data->[[mlm_input_token_ids, pred_positions, mlm_pred_labels],
        #            [],...] label是id
        return self.all_token_ids, self.attention_mask

    @staticmethod
    def encode(max_len, vocab_path, text_list):
        tokenizer = BertTokenizer.from_pretrained(vocab_path)
        inputs_ids = []

        for text in text_list:
            _tokenizer = tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=max_len,
                # return_tensors='pt'  # 返回的类型为pytorch tensor
            )
            input_ids = _tokenizer['input_ids']
            # attention_mask = _tokenizer['attention_mask']
            inputs_ids.append(input_ids)
        return inputs_ids

    @staticmethod
    def replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_pred,
                           vocab: Vocabulary):
        """
        Args:
            tokens: BERT输入序列的词元的列表 ids
            candidate_pred_positions: 不包括特殊词元的BERT输入序列的词元索引的列表（特殊词元在遮蔽语言模型任务中不被预测）
            num_mlm_pred: 预测的数量（选择15%要预测的随机词元）。
            vocab:
        Returns:
        """

        # 为遮蔽语言模型的输入创建新的词元副本
        mlm_input_tokens = [token for token in tokens]
        pred_positions_and_labels = []
        mask_id = vocab[b'[MASK]']

        random.shuffle(candidate_pred_positions)
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions_and_labels) >= num_mlm_pred:
                break
            masked_token = None
            # 80%将词替换为“[MASK]”
            if random.random() < 0.8:
                masked_token = mask_id
            else:
                # 10%保持词不变
                if random.random() < 0.5:
                    masked_token = tokens[mlm_pred_position]
                # 10%用随机词替换该词
                else:
                    masked_token = random.randint(0, len(vocab) - 1)
            mlm_input_tokens[mlm_pred_position] = masked_token
            pred_positions_and_labels.append(
                (mlm_pred_position, tokens[mlm_pred_position]))
        return mlm_input_tokens, pred_positions_and_labels

    def get_mlm_data(self, tokens, vocab: Vocabulary):
        candidate_pred_positions = []

        cls_id = self.vocab[b'[CLS]']
        sep_id = self.vocab[b'[SEP]']

        for i, token in enumerate(tokens):
            if token in [cls_id, sep_id]:
                continue
            candidate_pred_positions.append(i)

        # 15%的随机词元
        num_mlm_preds = max(1, round(len(tokens) * 0.15))
        # mlm_input_token是已经掩蔽过的句子
        mlm_input_token, pred_positions_and_labels = self.replace_mlm_tokens(
            tokens, candidate_pred_positions, num_mlm_preds, vocab)
        pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
        pred_positions = [v[0] for v in pred_positions_and_labels]

        # label是id
        mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
        return mlm_input_token, pred_positions, mlm_pred_labels

    @staticmethod
    def get_next_sentence(sentence, next_sentence, paragraphs):
        if random.random() < 0.5:
            is_next = True
        else:
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = False
        return sentence, next_sentence, is_next

    def get_nsp_data(self, paragraph, paragraphs, max_len):
        nsp_data_from_paragraph = []
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b, is_next = self.get_next_sentence(
                paragraph[i], paragraph[i + 1], paragraphs)

            if len(tokens_a) + len(tokens_b) - 1 > max_len:
                continue
            tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
            nsp_data_from_paragraph.append((tokens, segments, is_next))
        return nsp_data_from_paragraph

    @staticmethod
    def cut_sent(para):
        """
        将段落分成句子
        """
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
        para = para.rstrip()  # 段尾如果有多余的\n
        return para.split("\n")

    @staticmethod
    def pad_bert_inputs(examples, max_len, vocab):
        """
        Args:
            examples:
            [[mlm_input_token_ids, pred_positions, mlm_pred_labels,segments,is_next],
                   [],...]
            max_len:
            vocab:
        Returns:
        """
        max_num_mlm_preds = round(max_len * 0.15)
        all_token_ids, all_segments, valid_lens, mask = [], [], [], []
        all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
        nsp_labels = []
        pad_id = vocab['[PAD]']

        for (token_ids, pred_positions, mlm_pred_label_ids, segments,
             is_next) in examples:
            all_token_id = torch.tensor(token_ids + [pad_id] * (
                    max_len - len(token_ids)), dtype=torch.long)
            all_token_ids.append(all_token_id)

            # mask.append((torch.tensor( all_token_id!=pad_id, dtype=torch.long))

            all_segments.append(torch.tensor(segments + [0] * (
                    max_len - len(segments)), dtype=torch.long))
            # valid_lens不包括'[PAD]'的计数
            valid_len = len(token_ids)
            valid_lens.append(torch.tensor(valid_len, dtype=torch.float32))

            all_pred_positions.append(torch.tensor(pred_positions + [0] * (
                    max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
            # 填充词元的预测将通过乘以0权重在损失中过滤掉
            all_mlm_weights.append(
                torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                        max_num_mlm_preds - len(pred_positions)),
                             dtype=torch.float32))
            all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
                    max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
            nsp_labels.append(torch.tensor(is_next, dtype=torch.long))

            mask_att = [1.0] * valid_len + [0.0] * (max_len - valid_len)
            mask.append(torch.tensor(mask_att, dtype=torch.float32))

        return (all_token_ids, all_segments, valid_lens, all_pred_positions,
                all_mlm_weights, all_mlm_labels, nsp_labels, mask)


class EmoDataset(Dataset):
    def __init__(self, input_path, params, vocab_path, max_len):
        df = pd.read_csv(input_path)
        text_pd = df['微博中文内容'].astype('str')
        text_pd = text_pd.apply(self.remove_useless())
        text_list = list(text_pd)

        labels = list(df['情感倾向'].astype('int')+1)

        text_list, labels = self.remove_null(text_list, labels)

        self.input_ids = self.encode(max_len=max_len, vocab_path=vocab_path,
                                     text_list=text_list)
        self.labels = torch.tensor(labels)

        self.vocab = params.vocabulary['source']

        # self.examples=[ get_tokens_and_segments(example) for example in self.input_ids ]
        #
        # self.tokens, self.segments = get_tokens_and_segments(self.input_ids)

        # [(token,segment,label), ...]
        self.examples = list(zip(self.input_ids, self.labels))

        self.all_token_ids, self.valid_lens, self.mask, self.all_labels = self.pad_bert_inputs(
            self.examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return self.all_token_ids[idx], self.valid_lens[idx], \
               self.mask[idx], self.all_labels[idx]

    def __len__(self):
        return len(self.input_ids)

    @staticmethod
    def encode(max_len, vocab_path, text_list):
        # 将text_list embedding成bert模型可用的输入形式
        # 加载分词模型
        tokenizer = BertTokenizer.from_pretrained(vocab_path)
        tokenizer = tokenizer(
            text_list,
            padding=False,
            truncation=True,
            max_length=max_len,
            # return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        input_ids = tokenizer['input_ids']
        return input_ids

    @staticmethod
    def pad_bert_inputs(examples, max_len, vocab):
        """
        Args:
            examples:
             [(token,segment,label), ...]
            max_len:
            vocab:
        Returns:
        """

        all_token_ids, valid_lens, mask = [], [], []
        all_labels = []

        pad_id = vocab['[PAD]']

        for (token_ids, label) in examples:
            all_token_id = torch.tensor(token_ids + [pad_id] * (
                    max_len - len(token_ids)), dtype=torch.long)
            all_token_ids.append(all_token_id)

            valid_len = len(token_ids)
            valid_lens.append(torch.tensor(valid_len, dtype=torch.float32))

            mask_att = [1.0] * valid_len + [0.0] * (max_len - valid_len)
            mask.append(torch.tensor(mask_att, dtype=torch.float32))

            all_labels.append(torch.tensor(label, dtype=torch.long))

        return all_token_ids, valid_lens, mask, all_labels

    @staticmethod
    def remove_useless(text):
        # 去掉转发对象 回复对象
        rule1 = re.compile("//@.*:|回复@.*:")
        # 去掉?展开全文c O网页链接 ...
        rule2 = re.compile("\?展开全文c|O网页链接\?*|原标题：|转发微博|网易链接|查看图片")
        # 去掉 #,【】...
        rule3 = re.compile("[#【】]")
        text = rule1.sub(" ", text)
        text = rule2.sub(" ", text)
        text = rule3.sub(" ", text)
        text = text.strip()
        return text

    @staticmethod
    def remove_null(text_list, labels):
        _text_list = [item for item in text_list]
        for index, item in enumerate(text_list):
            if item.strip() == "":
                del _text_list[index]
                del labels[index]
        return _text_list, labels

```

bert.py 写了BERT的模型
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

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



```

emo_bert.py 情感分类模型
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

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

```
