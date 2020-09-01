from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

y_pred = []
with open('models/ner-from-tbaq-class-timex-bert-base-uncased-timebank-2/pred_results_timebank.txt', 'r') as reader:
    for line in reader:
        y_pred.append(line.strip('\n'))

y_true = []
with open('datasets/TBAQ-cleaned/ner-from-tbaq/class-v3/dev.txt', 'r') as reader:
    for line in reader:
        if not line.startswith('#'):
            items = line.split('\t')
            modality = items[3]
            if modality in ["DATE", "TIME", "DURATION", "SET"]:
                y_true.append(modality)

# print(y_pred)
print(len(y_pred))
# print(y_true)
print(len(y_true))

print(classification_report(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
