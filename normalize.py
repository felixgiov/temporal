import time_normalization as tn


f = open("datasets/MCTACO/dev_3783.tsv", "r")
lines = [x.strip() for x in f.readlines()]

normalized = []
for line in lines:
    new_line = ""
    group = line.split("\t")

    context = group[0]
    # for word in context:
    #     new_context = tn.quantity(tn, tokens=word)
    #     abc.append(new_context)

    # new_context = tn.normalize_timex(tn, expression=new_context)

    question = group[1]

    abc = ""
    answer = group[2]

    new_context = tn.quantity(tn, answer)
    # new_context2 = tn.normalize_timex(tn, answer)
    print(str(answer) + " " + str(new_context))
    # print(str(answer) + " " + str(new_context2))

    label = group[3]
    type = group[4]

    # new_line = abc
    # normalized.append(new_line)

# print(normalized)


