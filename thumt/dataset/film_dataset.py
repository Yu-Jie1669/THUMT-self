import random
import re

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

import thumt.data


class FilmDataset(Dataset):

    def __init__(self, input_path, params, vocab_path, max_len):
        with open(input_path, 'r', encoding='UTF-8') as f:
            paragraphs = f.readlines()
        # paragraphs ->[paragraph_count,sentences_count]
        paragraphs = [self.cut_sent(paragraph) for paragraph in paragraphs]
        # # all sentences
        # sentences = [sentence for paragraph in paragraphs for sentence in paragraph]

        self.vocab = params.vocabulary['source']

        self.input_ids= self.encode(max_len, vocab_path, paragraphs)
        # encode 出来前缀自带 [CLS] 后缀自带 [SEP]

        # 获取下一句子预测任务的数据
        examples = []

        # examples->[(tokens(id), segments, is_next) * len(example)]
        for paragraph_ids in self.input_ids:
            examples.extend(self.get_nsp_data(
                paragraph_ids, self.input_ids, max_len))

        # 获得MASK任务数据
        examples = [(self.get_mlm_data(tokens, self.vocab)
                     + (segments, is_next))
                    for tokens, segments, is_next in examples]

        # for i in range(len(examples)):
        #     examples[i][0] = self.input_ids[i]

        # examples->[[mlm_input_token_ids, pred_positions, mlm_pred_labels,segments,is_next],
        #            [],...]
        # label:id

        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels,self.attention_mask) = self.pad_bert_inputs(
            examples, max_len, self.vocab)


        # 填充输入
        # (self.all_token_ids, self.all_segments, self.valid_lens,
        #  self.all_pred_positions, self.all_mlm_weights,
        #  self.all_mlm_labels, self.nsp_labels) = BERTModel.pad_bert_inputs(
        #     examples, max_len, self.vocab)

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
                           vocab: thumt.data.Vocabulary):
        """
        Args:
            tokens: BERT输入序列的词元的列表 ids
            candidate_pred_positions: 不包括特殊词元的BERT输入序列的词元索引的列表（特殊词元在遮蔽语言模型任务中不被预测）
            num_mlm_pred: 预测的数量（选择15%要预测的随机词元）。
            vocab:
        Returns:
        """
        # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“[mask]”或随机词元
        mlm_input_tokens = [token for token in tokens]
        pred_positions_and_labels = []
        mask_id = vocab[b'[MASK]']
        # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
        random.shuffle(candidate_pred_positions)
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions_and_labels) >= num_mlm_pred:
                break
            masked_token = None
            # 80%将词替换为“[MASK]”词元
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

    def get_mlm_data(self, tokens, vocab: thumt.data.Vocabulary):
        candidate_pred_positions = []

        cls_id = self.vocab[b'[CLS]']
        sep_id = self.vocab[b'[SEP]']

        for i, token in enumerate(tokens):
            # 在遮蔽语言模型任务中不会预测特殊词元
            if token in [cls_id, sep_id]:
                continue
            candidate_pred_positions.append(i)

        # 遮蔽语言模型任务中预测15%的随机词元
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
            # paragraphs是三重列表的嵌套
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = False
        return sentence, next_sentence, is_next

    def get_tokens_and_segments(self, tokens_a, tokens_b=None):
        """获取输入序列的词元及其片段索引"""
        tokens = tokens_a
        # 0和1分别标记片段A和B
        segments = [0] * (len(tokens_a) )
        if tokens_b is not None:
            tokens += tokens_b[1:]
            segments += [1] * (len(tokens_b)-1)
        return tokens, segments

    def get_nsp_data(self, paragraph, paragraphs, max_len):
        nsp_data_from_paragraph = []
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b, is_next = self.get_next_sentence(
                paragraph[i], paragraph[i + 1], paragraphs)

            if len(tokens_a) + len(tokens_b) -1 > max_len:
                continue
            tokens, segments = self.get_tokens_and_segments(tokens_a, tokens_b)
            nsp_data_from_paragraph.append((tokens, segments, is_next))
        return nsp_data_from_paragraph

    @staticmethod
    def cut_sent(para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
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
                all_mlm_weights, all_mlm_labels, nsp_labels,mask)
