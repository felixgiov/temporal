import random

path = '/home/felix/projects/research/datasets/MCTACO/dev_3783.tsv'
path_new = '/home/felix/projects/research/datasets/MCTACO/cross_val/'

dictionary = {}
# with open(path, 'r') as reader:
#     for line in reader:
#         items = line.split("\t")
#         key = items[0]+"\t"+items[1]
#         value = items[2]+"\t"+items[3]+"\t"+items[4]
#         if key in dictionary:
#             dictionary[key].extend([value])
#         else:
#             dictionary[key] = []
#             dictionary[key].extend([value])

with open(path, 'r') as reader:
    for line in reader:
        items = line.split("\t")
        key = items[0]
        value = items[1]+"\t"+items[2]+"\t"+items[3]+"\t"+items[4]
        if key in dictionary:
            dictionary[key].extend([value])
        else:
            dictionary[key] = []
            dictionary[key].extend([value])

dev_size = int(0.2 * len(dictionary))
print(dev_size)

random.seed(21)
list = list(dictionary.items())
random.shuffle(list)

for i in range(0,5):
    train_1 = list[(i+1) * dev_size:]
    dev = list[i * dev_size:(i+1) * dev_size]
    train_2 = list[0:i * dev_size]

    print("Train A "+str(i)+" : "+str(len(train_1)))
    print("Train B "+str(i)+" : "+str(len(train_2)))
    print("Dev "+str(i)+" : "+str(len(dev)))

    with open(path_new+"train_"+str(i)+".tsv", 'w') as writer:
        for item in train_1:
            for answer in item[1]:
                writer.write(item[0] + "\t" + answer)
        for item in train_2:
            for answer in item[1]:
                writer.write(item[0] + "\t" + answer)

    with open(path_new+"dev_"+str(i)+".tsv", 'w') as writer:
        for item in dev:
            for answer in item[1]:
                writer.write(item[0] + "\t" + answer)

