import os

lines = []
# files_path = 'datasets/TBAQ-cleaned/TimeBank_joint/'
# files_path = 'datasets/TBAQ-cleaned/AQUAINT_joint/'
files_path = 'datasets/te3/te3-platinum_joint/'
for filename in os.listdir(files_path):
    with open(os.path.join(files_path, filename), 'r') as f:
        for line in f:
            lines.append(line)
    lines.append("\n")

# with open('datasets/TBAQ-cleaned/timebank_v2_max_index.txt', 'w') as w:
# with open('datasets/TBAQ-cleaned/aquaint.txt', 'w') as w:
with open('datasets/te3/te3-platinum.txt', 'w') as w:
    for line2 in lines:
        w.write(line2)
