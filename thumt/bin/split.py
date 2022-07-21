import random

film_path = "../../data/chinese/film.txt"

with open(film_path, 'r',encoding='UTF-8') as f:
    lines = f.readlines()

random.shuffle(lines)
lines=lines[:100000]

train_data=lines[:int(0.8*len(lines))]
dev_data=lines[int(0.8*len(lines)):int(0.9*len(lines))]
test_data=lines[int(0.9*len(lines)):]

with open("../../data/chinese/train_data.txt",'w',encoding='UTF-8') as f:
    f.writelines(train_data)

with open("../../data/chinese/dev_data.txt",'w',encoding='UTF-8') as f:
    f.writelines(dev_data)

with open("../../data/chinese/test_data.txt",'w',encoding='UTF-8') as f:
    f.writelines(test_data)