# import torch
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
# # roberta.eval()  # disable dropout (or leave in train mode to finetune)
#
# tokens = roberta.encode('Hello world!')
# print(tokens)
# assert tokens.tolist() == [0, 31414, 232, 328, 2]
# print(roberta.decode(tokens))  # 'Hello world!'

from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='MCTACO-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
preds = []
with open('../datasets/MCTACO/RTE-format/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        preds.append(prediction_label)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))

with open('../models/fairseq-roberta-large/preds.txt', 'w') as writer:
    for line in preds:
        writer.write(line+"\n")