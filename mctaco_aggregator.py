import os
import json

def extractResultsFromJSON(file_name):
    lines = []
    with open(file_name, 'r') as reader:
        for line in reader:
            lines.append(line)
    output = []
    em_f1_dev = []
    em_f1_test = []
    n = 3
    lines_splitted = [lines[i * n:(i + 1) * n] for i in range((len(lines) + n - 1) // n)]
    for item in lines_splitted:
        for j, line in enumerate(item):
            if j == 0:
                line_dict = json.loads(line)
                em_f1_dev.append((line_dict['eval_EM'], line_dict['eval_F1']))
            elif j == 1:
                line_dict = json.loads(line)
                em_f1_test.append((line_dict['eval_EM'], line_dict['eval_F1']))
    output.append(em_f1_dev)
    output.append(em_f1_test)

    return output


paths = ['/home/felix/projects/research/multi_results_1/',
         '/home/felix/projects/research/multi_results_2/',
         '/home/felix/projects/research/multi_results_3/']

exps = ['1', '12', '123', '1234', '12345', '1235', '124', '1245',
       '125', '13', '134', '1345', '135', '14', '145', '15']

# exps = ['1', '12', '123', '1234', '1235', '124', '1245',
#        '125', '13', '134', '1345', '135', '14', '145', '15']

# a1 = a12 = a123 = a1234 = a12345 = a1235 = a124 = a1245 = a125 = a13 = a134 = a1345 = a135 = a14 = a145 = a15 = []
# b1 = b12 = b123 = b1234 = b12345 = b1235 = b124 = b1245 = b125 = b13 = b134 = b1345 = b135 = b14 = b145 = b15 = []
# c1 = c12 = c123 = c1234 = c12345 = c1235 = c124 = c1245 = c125 = c13 = c134 = c1345 = c135 = c14 = c145 = c15 = []

# z1 = z12 = z123 = z1234 = z12345 = z1235 = z124 = z1245 = z125 = z13 = z134 = z1345 = z135 = z14 = z145 = z15 = []

dict = {}

for i, path in enumerate(paths):
    for exp in exps:
        directory = path + exp
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                if i == 0:
                    dict['a' + exp] = extractResultsFromJSON(directory+'/'+filename)
                elif i == 1:
                    dict['b' + exp] = extractResultsFromJSON(directory+'/'+filename)
                elif i == 2:
                    dict['c' + exp] = extractResultsFromJSON(directory+'/'+filename)

# print(dict['a1'][0][0][0])
# print(len(dict['a1']))
# print(len(dict['a1'][0]))
# print(len(dict['a1'][1]))

aggregated_dict = {}

for exp in exps:
    agg_list = []
    if len(dict['a' + exp][0]) == len(dict['a' + exp][1]) == len(dict['b' + exp][0]) == len(dict['b' + exp][1]) == len(dict['c' + exp][0]) == len(dict['c' + exp][1]):
        for i in range(len(dict['a' + exp][0])):
            dev_em_avg = (dict['a' + exp][0][i][0] + dict['b' + exp][0][i][0] + dict['c' + exp][0][i][0]) / 3
            dev_f1_avg = (dict['a' + exp][0][i][1] + dict['b' + exp][0][i][1] + dict['c' + exp][0][i][1]) / 3
            test_em_avg = (dict['a' + exp][1][i][0] + dict['b' + exp][1][i][0] + dict['c' + exp][1][i][0]) / 3
            test_f1_avg = (dict['a' + exp][1][i][1] + dict['b' + exp][1][i][1] + dict['c' + exp][1][i][1]) / 3
            agg_tuple = (dev_em_avg, dev_f1_avg, test_em_avg, test_f1_avg)
            agg_list.append(agg_tuple)
        aggregated_dict['z' + exp] = agg_list

# print(aggregated_dict['z15'][8])
best_list = []

for exp in exps:
    max_f1_dev = -99
    max_idx = -1
    for i, tuple_item in enumerate(aggregated_dict['z' + exp]):
        if tuple_item[1] > max_f1_dev:
            max_idx = i
            max_f1_dev = tuple_item[1]
    best_dev_em = aggregated_dict['z' + exp][max_idx][0]
    best_dev_f1 = aggregated_dict['z' + exp][max_idx][1]
    best_test_em = aggregated_dict['z' + exp][max_idx][2]
    best_test_f1 = aggregated_dict['z' + exp][max_idx][3]
    best_list.append((exp, max_idx+1, '%.4f' % best_dev_em, '%.4f' % best_dev_f1, '%.4f' % best_test_em, '%.4f' % best_test_f1))

print(best_list)
