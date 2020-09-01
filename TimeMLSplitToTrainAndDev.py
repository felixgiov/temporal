import math
import random

sentences=[]
sentence=[]
with open('datasets/TBAQ-cleaned/timebank_v4.txt', 'r') as reader:
    for line in reader:
        if line.startswith("#"):
            sentence=[]
            sentences.append(sentence)
            sentence.append(line)
        else:
            sentence.append(line)

# with open('datasets/TBAQ-cleaned/aquaint_v2.txt', 'r') as reader:
#     for line in reader:
#         if line.startswith("#"):
#             sentence=[]
#             sentences.append(sentence)
#             sentence.append(line)
#         else:
#             sentence.append(line)

random.seed(21)
random.shuffle(sentences)
train_data = sentences[:math.floor(0.9*len(sentences))]
dev_data = sentences[math.floor(0.9*len(sentences)):]

with open("datasets/TBAQ-cleaned/timebank_v4_train.txt", "w") as writer:
    for line1 in train_data:
        for line2 in line1:
            writer.write(line2)

with open("datasets/TBAQ-cleaned/timebank_v4_dev.txt", "w") as writer:
    for line1 in dev_data:
        for line2 in line1:
            writer.write(line2)