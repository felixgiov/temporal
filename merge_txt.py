import os

lines = []
files_path = 'datasets/TBAQ-cleaned/TimeBank_joint/'
for filename in os.listdir(files_path):
    with open(os.path.join(files_path, filename), 'r') as f:
        for line in f:
            lines.append(line)
    lines.append("\n")

with open('datasets/TBAQ-cleaned/timebank.txt', 'w') as w:
    for line2 in lines:
        w.write(line2)
