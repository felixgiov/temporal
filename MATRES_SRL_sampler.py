import random

train_path = '/home/felix/projects/research/datasets/MATRES-MCTACO/train_srl.tsv'
test_path = '/home/felix/projects/research/datasets/MATRES-MCTACO/test_srl.tsv'

dictionary = {}

with open(test_path, 'r') as reader:
    for row in reader:
        item = row.split('\t')
        key = item[0]
        if key not in dictionary.keys():
            dictionary[key] = {}
            dictionary[key][item[1]] = [item[2] + '\t' + item[3] + '\t' + item[4]]
        else:
            if item[1] in dictionary[key].keys():
                dictionary[key][item[1]].append(item[2] + '\t' + item[3] + '\t' + item[4])
            else:
                dictionary[key][item[1]] = [item[2] + '\t' + item[3] + '\t' + item[4]]

sampled = []

for key in dictionary:
    rand_key = random.choice(dictionary[key].keys())
    for item in dictionary[key][rand_key]:
        sampled.append(key+'\t'+rand_key+'\t'+item)

with open('/home/felix/projects/research/datasets/MATRES-MCTACO/test_srl_sampled.tsv', 'w') as writer:
    for row in sampled:
        writer.write(row)


