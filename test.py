from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch import nn
import numpy as np

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = RobertaForSequenceClassification.from_pretrained('roberta-base')
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids, labels=labels)
# loss, logits = outputs[:2]
#
# print(input_ids)
# print(labels)
# print(outputs)
# print(loss)
# print(logits)

# m = nn.Softmax(dim=1)

# input = torch.randn(5,2)
# input = np.random.rand(5,2)
# input_tensor = torch.from_numpy(input)
#
# output = m(input_tensor)
#
# print(input)
# print(input_tensor)
# print(output)
#
# for val in output:
#     if val[0] > val[1]:
#         print("yes")
#     else:
#         print("no")
