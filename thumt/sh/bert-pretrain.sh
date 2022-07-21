#conda init
#
#conda activate py37

python ../bin/chinese_bert.py \
        --input ../../data/chinese/train_data.txt \
        --output bert-train \
        --vocabulary /home/linyujie/THUMT-self/data/chinese/vocab.txt\
        --model bert \
        --validation ../../data/chinese/dev_data.txt \
        --parameters=batch_size=512,update_cycle=2 \
        --hparam_set base\
        > bert-pretrain.log