#### 文件目录


    ├─bert-train
    ├─data
    │  ├─chinese 预训练数据 （电影评论） 词表
    │  ├─emo 情感分析数据
    ├─thumt
    │  ├─bin
    │  │  └─bert_pretrain.py BERT预训练
    │  │  └─emo_train.py 情感分析训练
    │  ├─data
    │  │  └─samples.py 电影Dataset（BERT的mask和nsps数据） 情感Dataset
    │  └─models
    │     └─bert.py BERT模型
    │     └─emo_bert.py 情感分类模型
    └─train