from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

y_pred = []
# with open('models/multi-mctaco-ner-roberta-large-2/pred_results_matres.txt', 'r') as reader:
# with open('models/multi-mctaco-ner-roberta-large-2/pred_results_matres_test.txt', 'r') as reader:
# with open('models/multi-mctaco-ner-roberta-large/pred_results_roberta-large.txt', 'r') as reader:
with open('models/multi-mctaco-ner-roberta-large/pred_results_matres_test.txt', 'r') as reader:
    for line in reader:
        y_pred.append(line.strip('\n'))

y_true = []
# with open('datasets/MATRES/eval_labels.txt', 'r') as reader:
with open('datasets/MATRES/test_labels.txt', 'r') as reader:
    for line in reader:
        y_true.append(line.strip("\n"))

# print(y_pred)
print(len(y_pred))
# print(y_true)
print(len(y_true))

print(classification_report(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
