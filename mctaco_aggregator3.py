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

def agg(lr, batch, seeds, dir_path):
    paths = []
    for seed in seeds:
        paths.append(dir_path + str(seed) + '_' + str(lr) + 'e-5_8x' + str(batch) + '/')

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
                    # print(directory)

    aggregated_dict = {}
    non_agg_dict = {}

    for exp in exps:
        agg_list, non_agg_list_a, non_agg_list_b, non_agg_list_c, non_agg_list_d, non_agg_list_e, \
            non_agg_list_f, non_agg_list_g, non_agg_list_h, non_agg_list_i, non_agg_list_j,\
            non_agg_list_k, non_agg_list_l, non_agg_list_m, non_agg_list_n, non_agg_list_o, = [], [], [], [], [], [],\
                                                                                              [], [], [], [], [], \
                                                                                              [], [], [], [], []
        if len(dict['a' + exp][0]) == len(dict['a' + exp][1]) == len(dict['b' + exp][0]) == len(dict['b' + exp][1]) == len(dict['c' + exp][0]) == len(dict['c' + exp][1]):
            for i in range(len(dict['a' + exp][0])):
                dev_em_avg = (dict['a' + exp][0][i][0] + dict['b' + exp][0][i][0] + dict['c' + exp][0][i][0]) / 3
                dev_f1_avg = (dict['a' + exp][0][i][1] + dict['b' + exp][0][i][1] + dict['c' + exp][0][i][1]) / 3
                test_em_avg = (dict['a' + exp][1][i][0] + dict['b' + exp][1][i][0] + dict['c' + exp][1][i][0]) / 3
                test_f1_avg = (dict['a' + exp][1][i][1] + dict['b' + exp][1][i][1] + dict['c' + exp][1][i][1])/ 3
                agg_tuple = (dev_em_avg, dev_f1_avg, test_em_avg, test_f1_avg)
                a_tuple = (
                dict['a' + exp][0][i][0], dict['a' + exp][0][i][1], dict['a' + exp][1][i][0], dict['a' + exp][1][i][1])
                b_tuple = (
                dict['b' + exp][0][i][0], dict['b' + exp][0][i][1], dict['b' + exp][1][i][0], dict['b' + exp][1][i][1])
                c_tuple = (
                dict['c' + exp][0][i][0], dict['c' + exp][0][i][1], dict['c' + exp][1][i][0], dict['c' + exp][1][i][1])
                agg_list.append(agg_tuple)
                non_agg_list_a.append(a_tuple)
                non_agg_list_b.append(b_tuple)
                non_agg_list_c.append(c_tuple)
            aggregated_dict['z' + exp] = agg_list
            non_agg_dict['a' + exp] = non_agg_list_a
            non_agg_dict['b' + exp] = non_agg_list_b
            non_agg_dict['c' + exp] = non_agg_list_c
        else:
            print("Difference in lengths in exp "+exp)
            print(str(len(dict['a' + exp][0])) +" "+str(len(dict['a' + exp][1])) +" "+str(len(dict['b' + exp][0])) +" "+
                  str(len(dict['b' + exp][1])) +" "+str(len(dict['c' + exp][0])) +" "+str(len(dict['c' + exp][1])))

    print(getBest(aggregated_dict, 'z', 0, 10, 0))

if __name__ == '__main__':
    # seeds = ['31', '32', '33']
    # lrs = ['1', '2', '5']
    # batchs = ['4', '8', '16']
    # paths = []
    #
    # for seed in seeds:
    #     for lr in lrs:
    #         for batch in batchs:
    #             paths.append('/home/felix/projects/research/multi_results/')

    # seeds = ['31', '32', '33']
    seeds = ['33', '10321', '76567']
    lrs = [1, 2, 3]
    batchs = [2, 4, 8]

    # exps = ['1', '12', '123', '1234', '12345', '1235', '124', '1245',
    #        '125', '13', '134', '1345', '135', '14', '145', '15']

    # exps = ['1', '12', '13', '14', '15']

    exps = ['13', '14', '15']

    # dir_path = '/larch/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed'
    dir_path = '/orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed'

    for lr in lrs:
        for batch in batchs:
            print("LR: {} | GRAD: {}".format(lr, batch))
            agg(lr, batch,  seeds, dir_path)


