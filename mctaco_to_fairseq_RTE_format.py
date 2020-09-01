rows = []
rows.append("index\tsentence1\tsentence2\tlabel\n")
index = -1
with open('datasets/MCTACO/test_9442.tsv', 'r') as reader:
    for line in reader:
        index += 1
        items = line.split("\t")
        row = str(index)+"\t"+items[0]+" "+items[1]+"\t"+items[2]+"\t"+items[3]+"\n"
        rows.append(row)

with open("datasets/MCTACO/RTE-format/dev.tsv", "w") as writer:
    for line1 in rows:
        writer.write(line1)