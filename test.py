from transformers import BertTokenizer
from timebank_modeling_bert import BertForTokenEventClassification, BertConfig
import torch

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenEventClassification.from_pretrained('bert-base-uncased', config=config)

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
labels = torch.tensor([-100, 0, 0, 0, 1, 0, 2, -100]).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)

loss, scores = outputs[:2]

print(tokenizer.tokenize("Hello, my dog is cute"))
print(tokenizer.encode("Hello, my dog is cute", add_special_tokens=False))
print(input_ids)
print(torch.tensor([1] * input_ids.size(1)))
print(labels)
print(loss)
print(scores)

# torch.Size([1, 8, 768])                                                           sequence output
# tensor([[[-0.1144,  0.1937,  0.1250,  ..., -0.3827,  0.2107,  0.5407],
#          [ 0.5308,  0.3207,  0.3665,  ..., -0.0036,  0.7579,  0.0388],
#          [-0.4877,  0.8849,  0.4256,  ..., -0.6976,  0.4458,  0.1231],
#          ...,
#          [-0.7003, -0.1815,  0.3297,  ..., -0.4838,  0.0680,  0.8901],
#          [-1.0355, -0.2567, -0.0317,  ...,  0.3197,  0.3999,  0.1795],
#          [ 0.6080,  0.2610, -0.3131,  ...,  0.0311, -0.6283, -0.1994]]],
#        grad_fn=<NativeLayerNormBackward>)
# torch.Size([1, 8, 768])                                                          sequence output after dropout
# tensor([[[-0.1144,  0.1937,  0.1250,  ..., -0.3827,  0.2107,  0.5407],
#          [ 0.5308,  0.3207,  0.3665,  ..., -0.0036,  0.7579,  0.0388],
#          [-0.4877,  0.8849,  0.4256,  ..., -0.6976,  0.4458,  0.1231],
#          ...,
#          [-0.7003, -0.1815,  0.3297,  ..., -0.4838,  0.0680,  0.8901],
#          [-1.0355, -0.2567, -0.0317,  ...,  0.3197,  0.3999,  0.1795],
#          [ 0.6080,  0.2610, -0.3131,  ...,  0.0311, -0.6283, -0.1994]]],
#        grad_fn=<NativeLayerNormBackward>)
# torch.Size([1, 8, 3])                                                            logits
# tensor([[[ 0.9780, -0.1662, -0.1183],
#          [ 0.2666,  0.3688,  0.0591],
#          [ 0.1719, -0.0542, -0.0373],
#          [ 0.2429,  0.0533,  0.4143],
#          [ 0.0219, -0.0160,  0.1718],
#          [-0.6171,  0.1116,  0.4703],
#          [-0.0149,  0.4675,  0.7406],
#          [-0.0251, -0.2628,  0.5650]]], grad_fn=<AddBackward0>)
# ['hello', ',', 'my', 'dog', 'is', 'cute']
# [7592, 1010, 2026, 3899, 2003, 10140]
# tensor([[  101,  7592,  1010,  2026,  3899,  2003, 10140,   102]])
# tensor([1, 1, 1, 1, 1, 1, 1, 1])
# tensor([[-100,    0,    0,    0,    1,    0,    2, -100]])
# tensor(1.1519, grad_fn=<NllLossBackward>)
# tensor([[[ 0.9780, -0.1662, -0.1183],
#          [ 0.2666,  0.3688,  0.0591],
#          [ 0.1719, -0.0542, -0.0373],
#          [ 0.2429,  0.0533,  0.4143],
#          [ 0.0219, -0.0160,  0.1718],
#          [-0.6171,  0.1116,  0.4703],
#          [-0.0149,  0.4675,  0.7406],
#          [-0.0251, -0.2628,  0.5650]]], grad_fn=<AddBackward0>)


# samples = torch.cat((-torch.ones(5), torch.ones(5)))
# samples2 = torch.cat((torch.ones(50) * 5, torch.ones(5) * -5))
#
# print(samples)
# print(samples2)