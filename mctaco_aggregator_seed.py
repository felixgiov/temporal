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


def getBest(dict, type, min_epoch, max_epoch, criteria):
    best_list = []

    reduced_dict = {}
    for key in dict:
        reduced_dict[key] = dict[key][min_epoch:max_epoch]

    for exp in exps:
        max_f1_dev = -99
        max_idx = -1
        for i, tuple_item in enumerate(reduced_dict[type + exp]):
            if tuple_item[criteria] > max_f1_dev:
                max_idx = i
                max_f1_dev = tuple_item[criteria]
        best_dev_em = reduced_dict[type + exp][max_idx][0]
        best_dev_f1 = reduced_dict[type + exp][max_idx][1]
        best_test_em = reduced_dict[type + exp][max_idx][2]
        best_test_f1 = reduced_dict[type + exp][max_idx][3]
        best_list.append((exp, max_idx + 1, '%.4f' % best_dev_em, '%.4f' % best_dev_f1, '%.4f' % best_test_em,
                          '%.4f' % best_test_f1))

    return best_list


# seeds = ['31', '32', '33']
# lrs = ['1', '2', '5']
# batchs = ['4', '8', '16']
# paths = []
#
# for seed in seeds:
#     for lr in lrs:
#         for batch in batchs:
#             paths.append('/home/felix/projects/research/multi_results/')

seeds = ['31']
lr = 1
batch = 4

paths = []
for seed in seeds:
    for j in range(5):
        paths.append('/home/felix/projects/research/multi_results/multi_results_seed'+str(seed)+'_'+str(lr)+'e-5_8x'+str(batch)+'_'+str(j)+'/')

for path in paths:
    print(path)
print(len(paths))

# paths = ['/home/felix/projects/research/multi_results_1/',
#          '/home/felix/projects/research/multi_results_2/',
#          '/home/felix/projects/research/multi_results_3/',
#          '/home/felix/projects/research/multi_results_4/',
#          '/home/felix/projects/research/multi_results_5/']

# exps = ['1', '12', '123', '1234', '12345', '1235', '124', '1245',
#        '125', '13', '134', '1345', '135', '14', '145', '15']

exps = ['1', '12', '14']

# exps = ['1', '12', '123', '1234', '1235', '124', '1245',
#        '125', '13', '134', '1345', '135', '14', '145', '15']

dict = {}

for i, path in enumerate(paths):
    for exp in exps:
        directory = path + exp
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename == "results.txt":
                if i == 0:
                    dict['a' + exp] = extractResultsFromJSON(directory + '/' + filename)
                elif i == 1:
                    dict['b' + exp] = extractResultsFromJSON(directory + '/' + filename)
                elif i == 2:
                    dict['c' + exp] = extractResultsFromJSON(directory + '/' + filename)
                elif i == 3:
                    dict['d' + exp] = extractResultsFromJSON(directory + '/' + filename)
                elif i == 4:
                    dict['e' + exp] = extractResultsFromJSON(directory + '/' + filename)
                # print(directory)

# for i, path in enumerate(paths):
#     for exp in exps:
#         directory = path + exp
#         for file in os.listdir(directory):
#             filename = os.fsdecode(file)
#             if filename == "results.txt":
#                 dict[str(i) + exp] = extractResultsFromJSON(directory + '/' + filename)

# print(dict['a1'][0][0][0])
# print(len(dict['a1']))
# print(len(dict['a1'][0]))
# print(len(dict['a1'][1]))

aggregated_dict = {}
non_agg_dict = {}

for exp in exps:
    agg_list, non_agg_list_a, non_agg_list_b, non_agg_list_c, non_agg_list_d, non_agg_list_e = [], [], [], [], [], []
    if len(dict['a' + exp][0]) == 20:
        for i in range(len(dict['a' + exp][0])):
            dev_em_avg = (dict['a' + exp][0][i][0] + dict['b' + exp][0][i][0] + dict['c' + exp][0][i][0] +
                          dict['d' + exp][0][i][0] + dict['e' + exp][0][i][0]) / 5
            dev_f1_avg = (dict['a' + exp][0][i][1] + dict['b' + exp][0][i][1] + dict['c' + exp][0][i][1] +
                          dict['d' + exp][0][i][1] + dict['e' + exp][0][i][1]) / 5
            test_em_avg = (dict['a' + exp][1][i][0] + dict['b' + exp][1][i][0] + dict['c' + exp][1][i][0] +
                           dict['d' + exp][1][i][0] + dict['e' + exp][1][i][0]) / 5
            test_f1_avg = (dict['a' + exp][1][i][1] + dict['b' + exp][1][i][1] + dict['c' + exp][1][i][1] +
                           dict['d' + exp][1][i][1] + dict['e' + exp][1][i][1])/ 5
            agg_tuple = (dev_em_avg, dev_f1_avg, test_em_avg, test_f1_avg)
            a_tuple = (
            dict['a' + exp][0][i][0], dict['a' + exp][0][i][1], dict['a' + exp][1][i][0], dict['a' + exp][1][i][1])
            b_tuple = (
            dict['b' + exp][0][i][0], dict['b' + exp][0][i][1], dict['b' + exp][1][i][0], dict['b' + exp][1][i][1])
            c_tuple = (
            dict['c' + exp][0][i][0], dict['c' + exp][0][i][1], dict['c' + exp][1][i][0], dict['c' + exp][1][i][1])
            d_tuple = (
            dict['d' + exp][0][i][0], dict['d' + exp][0][i][1], dict['d' + exp][1][i][0], dict['d' + exp][1][i][1])
            e_tuple = (
            dict['e' + exp][0][i][0], dict['e' + exp][0][i][1], dict['e' + exp][1][i][0], dict['e' + exp][1][i][1])
            agg_list.append(agg_tuple)
            non_agg_list_a.append(a_tuple)
            non_agg_list_b.append(b_tuple)
            non_agg_list_c.append(c_tuple)
            non_agg_list_d.append(d_tuple)
            non_agg_list_e.append(e_tuple)
        aggregated_dict['z' + exp] = agg_list
        non_agg_dict['a' + exp] = non_agg_list_a
        non_agg_dict['b' + exp] = non_agg_list_b
        non_agg_dict['c' + exp] = non_agg_list_c
        non_agg_dict['d' + exp] = non_agg_list_d
        non_agg_dict['e' + exp] = non_agg_list_e
    else:
        print("Difference in lengths in exp "+exp)
        print(str(len(dict['a' + exp][0])) +" "+str(len(dict['a' + exp][1])) +" "+str(len(dict['b' + exp][0])) +" "+
              str(len(dict['b' + exp][1])) +" "+str(len(dict['c' + exp][0])) +" "+str(len(dict['c' + exp][1])) +" "+
              str(len(dict['d' + exp][0])) +" "+str(len(dict['d' + exp][1])) +" "+str(len(dict['e' + exp][0])) +" "+
              str(len(dict['e' + exp][1])))

# print(aggregated_dict['z15'][8])

# best_list = []
#
# reduced_aggregated_dict = {}
# for key in aggregated_dict:
#         reduced_aggregated_dict[key] = aggregated_dict[key][:30]
#
# for exp in exps:
#     max_f1_dev = -99
#     max_idx = -1
#     for i, tuple_item in enumerate(reduced_aggregated_dict['z' + exp]):
#         if tuple_item[0] > max_f1_dev:
#             max_idx = i
#             max_f1_dev = tuple_item[0]
#     best_dev_em = aggregated_dict['z' + exp][max_idx][0]
#     best_dev_f1 = aggregated_dict['z' + exp][max_idx][1]
#     best_test_em = aggregated_dict['z' + exp][max_idx][2]
#     best_test_f1 = aggregated_dict['z' + exp][max_idx][3]
#     best_list.append((exp, max_idx+1, '%.4f' % best_dev_em, '%.4f' % best_dev_f1, '%.4f' % best_test_em, '%.4f' % best_test_f1))
#
# print(best_list)

# print(non_agg_dict['a15'][8])
# print(aggregated_dict['z14'])
print(getBest(aggregated_dict, 'z', 0, 20, 0))
print(getBest(aggregated_dict, 'z', 0, 20, 2))

# print(getBest(non_agg_dict, 'a', 21, 22, 2))
# print(getBest(non_agg_dict, 'b', 21, 22, 2))
# print(getBest(non_agg_dict, 'c', 21, 22, 2))
# print(getBest(non_agg_dict, 'd', 21, 22, 2))
# print(getBest(non_agg_dict, 'e', 21, 22, 2))
