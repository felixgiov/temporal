def mctaco_evaluator(prediction_lines):

    test_file = '/home/felix/projects/research/datasets/MCTACO/test_9442.tsv'

    ref_lines = [x.strip() for x in open(test_file).readlines()]

    result_map = {}
    prediction_count_map = {}
    prediction_map = {}
    gold_count_map = {}
    for i, line in enumerate(ref_lines):
        key = " ".join(line.split("\t")[0:2])
        if key not in result_map:
            result_map[key] = []
            prediction_count_map[key] = 0.0
            gold_count_map[key] = 0.0
            prediction_map[key] = []
        prediction = prediction_lines[i]
        prediction_map[key].append(prediction)
        label = line.split("\t")[3]
        if prediction == "yes":
            prediction_count_map[key] += 1.0
        if label == "yes":
            gold_count_map[key] += 1.0
        result_map[key].append(prediction == label)

    total = 0.0
    correct = 0.0
    f1 = 0.0
    for question in result_map:
        val = True
        total += 1.0
        cur_correct = 0.0
        for i, v in enumerate(result_map[question]):
            val = val and v
            if v and prediction_map[question][i] == "yes":
                cur_correct += 1.0
        if val:
            correct += 1.0
        p = 1.0
        if prediction_count_map[question] > 0.0:
            p = cur_correct / prediction_count_map[question]
        r = 1.0
        if gold_count_map[question] > 0.0:
            r = cur_correct / gold_count_map[question]
        if p + r > 0.0:
            f1 += 2 * p * r / (p + r)

    em = correct / total
    f1 = f1 / total
    # avg = (em + f1) / 2

    # print("Avg: {:.4f}".format(avg))
    # print("EM: {:.4f}".format(em))
    # print("F1: {:.4f}".format(f1))

    return [em, f1]

def most_frequent(List):
    return max(set(List), key=List.count)

def agg(seeds, lr, grad, epoch, comb, dir_path):
    paths = []

    # for seed in seeds:
    #     for lr in lrs:
    #         for grad in grads:
    #             paths.append(dir_path+'multi_results_seed'+seed+'_'+lr+'e-5_8x'+grad+'/'+comb+'/'+epoch)

    for seed in seeds:
        paths.append(dir_path+'multi_results_seed'+seed+'_'+lr+'e-5_8x'+grad+'/'+comb+'/'+str(epoch))

    result_dict = {}
    agg_result = []

    for i, path in enumerate(paths):
        with open(path+'/pred_results.txt', "r") as reader:
            for j, row in enumerate(reader):
                ans = str(row).strip("\n")
                if ans == 'yes' or ans == 'no':
                    result_dict[seeds[i] + '_' + str(j)] = ans

    for k in range(9442):
        first_ans = result_dict[seeds[0] + '_' + str(k)]
        second_ans = result_dict[seeds[1] + '_' + str(k)]
        third_ans = result_dict[seeds[2] + '_' + str(k)]
        results = [first_ans, second_ans, third_ans]
        if most_frequent(results) == 'yes':
            agg_result.append('yes')
        else:
            agg_result.append('no')

    y = mctaco_evaluator(agg_result)

    return y

def agg_five(seeds, lr, grad, dir_path):
    paths = []

    for seed in seeds:
        paths.append(dir_path+'_result_ensemble_logits_seed'+seed+'_'+lr+'e-5_8x'+grad+'_10')

    result_dict = {}
    agg_result = []
    first_ans_list = []
    second_ans_list = []
    third_ans_list = []

    for i, path in enumerate(paths):
        with open(path+'/pred_results.txt', "r") as reader:
            for j, row in enumerate(reader):
                ans = str(row).strip("\n")
                if ans == 'yes' or ans == 'no':
                    result_dict[seeds[i] + '_' + str(j)] = ans

    for k in range(9442):
        first_ans = result_dict[seeds[0] + '_' + str(k)]
        second_ans = result_dict[seeds[1] + '_' + str(k)]
        third_ans = result_dict[seeds[2] + '_' + str(k)]

        first_ans_list.append(first_ans)
        second_ans_list.append(second_ans)
        third_ans_list.append(third_ans)

        results = [first_ans, second_ans, third_ans]
        if most_frequent(results) == 'yes':
            agg_result.append('yes')
        else:
            agg_result.append('no')
        print(results)

    y = mctaco_evaluator(agg_result)
    z1 = mctaco_evaluator(first_ans_list)
    z2 = mctaco_evaluator(second_ans_list)
    z3 = mctaco_evaluator(third_ans_list)

    return y, z1, z2, z3

def agg_multi(seeds, lr, grad, epoch, combs, dir_path):
    paths = []

    for comb in combs:
        for seed in seeds:
            paths.append(dir_path+'multi_results_seed'+seed+'_'+lr+'e-5_8x'+grad+'/'+comb+'/'+str(epoch))

    result_dict = {}
    agg_result = []

    for i, path in enumerate(paths):
        with open(path+'/pred_results.txt', "r") as reader:
            for j, row in enumerate(reader):
                ans = str(row).strip("\n")
                if ans == 'yes' or ans == 'no':
                    result_dict[str(i) + '_' + str(j)] = ans

    for k in range(9442):
        ans1 = result_dict['0_' + str(k)]
        ans2 = result_dict['1_' + str(k)]
        ans3 = result_dict['2_' + str(k)]
        ans4 = result_dict['3_' + str(k)]
        ans5 = result_dict['4_' + str(k)]
        ans6 = result_dict['5_' + str(k)]
        # ans7 = result_dict['6_' + str(k)]
        # ans8 = result_dict['7_' + str(k)]
        # ans9 = result_dict['8_' + str(k)]
        # ans10 = result_dict['9_' + str(k)]
        # ans11 = result_dict['10_' + str(k)]
        # ans12 = result_dict['11_' + str(k)]
        # ans13 = result_dict['12_' + str(k)]
        # ans14 = result_dict['13_' + str(k)]
        # ans15 = result_dict['14_' + str(k)]
        # results = [ans1, ans2, ans3, ans4, ans5, ans6, ans7, ans8, ans9, ans10, ans11, ans12, ans13, ans14, ans15]
        results = [ans1, ans2, ans3, ans4, ans5, ans6]
        if most_frequent(results) == 'yes':
            agg_result.append('yes')
        else:
            agg_result.append('no')

    y = mctaco_evaluator(agg_result)

    return y

def agg_multi_specific(seeds, dir_path):
    paths = []

    # for specific epoch
    for seed in seeds:
        paths.append(dir_path + 'multi_results_seed' + seed + '_1e-5_8x4/1/10')
        # paths.append(dir_path + 'multi_results_seed' + seed + '_1e-5_8x8/12/10')
        # paths.append(dir_path + 'multi_results_seed' + seed + '_1e-5_8x2/13/10')
        # paths.append(dir_path + 'multi_results_seed' + seed + '_1e-5_8x8/14/10')
        # paths.append(dir_path + 'multi_results_seed' + seed + '_1e-5_8x4/15/10')

    result_dict = {}
    agg_result = []

    for i, path in enumerate(paths):
        with open(path+'/pred_results.txt', "r") as reader:
            for j, row in enumerate(reader):
                ans = str(row).strip("\n")
                if ans == 'yes' or ans == 'no':
                    result_dict[str(i) + '_' + str(j)] = ans

    for k in range(9442):
        ans1 = result_dict['0_' + str(k)]
        ans2 = result_dict['1_' + str(k)]
        ans3 = result_dict['2_' + str(k)]
        ans4 = result_dict['3_' + str(k)]
        ans5 = result_dict['4_' + str(k)]
        ans6 = result_dict['5_' + str(k)]
        ans7 = result_dict['6_' + str(k)]
        ans8 = result_dict['7_' + str(k)]
        ans9 = result_dict['8_' + str(k)]
        ans10 = result_dict['9_' + str(k)]
        ans11 = result_dict['10_' + str(k)]
        ans12 = result_dict['11_' + str(k)]
        ans13 = result_dict['12_' + str(k)]
        ans14 = result_dict['13_' + str(k)]
        ans15 = result_dict['14_' + str(k)]
        results = [ans1, ans2, ans3, ans4, ans5, ans6, ans7, ans8, ans9, ans10, ans11, ans12, ans13, ans14, ans15]
        if most_frequent(results) == 'yes':
            agg_result.append('yes')
        else:
            agg_result.append('no')

    output_pred_file = "/orange/felix/multi_results/ensemble_vote_1_15models_pred_results.txt"

    with open(output_pred_file, "w") as pred_writer:
        for row in agg_result:
            pred_writer.write(row+"\n")

    y = mctaco_evaluator(agg_result)

    return y

if __name__ == '__main__':
    # seeds = ['33', '10321', '76567']
    seeds = ['33', '10321', '76567', '843029', '5675433', '93430', '5435', '64829', '389239', '435854', '7450', '9953', '24243', '4862', '234848']
    lrs = ['1', '2', '3']
    grads = ['2', '4', '8']
    comb = '15'
    # combs = ['1', '12', '13', '14', '15']
    combs = ['1']
    total_epoch = 10

    # 2e = old cosmos , 3e new cosmos
    # dir_path = '/home/felix/projects/research/multi_results/multi_run_cosmos_new_3e/'
    # dir_path = '/larch/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/'
    dir_path = '/orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/'
    # dir_path = '/larch/felix/multi_results/multi_run_cosmos_new_3e_default_dropout_warmup2/'

    # for lr in lrs:
    #     for grad in grads:
    #         print("LR: {} | GRAD: {}".format(lr, grad))
    #         result_dict = {}
    #         for epoch in range(10,total_epoch+1):
    #             result_dict[epoch] = agg_multi(seeds, lr, grad, epoch, combs, dir_path)
    #         max_key = -1
    #         max_em = -1
    #         for key in result_dict:
    #             if result_dict[key][0] > max_em:
    #                 max_em = result_dict[key][0]
    #                 max_key = key
    #         print("\tBEST EPOCH = {} | EM: {:.4f} | F1: {:.4f}".format(max_key, result_dict[max_key][0], result_dict[max_key][1]))

    em_s, f1_s = agg_multi_specific(seeds, dir_path)
    print("====\nEM: {:.4f} | F1: {:.4f}".format(em_s, f1_s))


    # for lr in lrs:
    #     for grad in grads:
    #         print("LR: {} | GRAD: {}".format(lr, grad))
    #         a, b, c, d = agg_five(seeds, lr, grad, dir_path)
    #         print("\tSeed {} | EM: {:.4f} | F1: {:.4f}".format(seeds[0], b[0], b[1]))
    #         print("\tSeed {} | EM: {:.4f} | F1: {:.4f}".format(seeds[1], c[0], c[1]))
    #         print("\tSeed {} | EM: {:.4f} | F1: {:.4f}".format(seeds[2], d[0], d[1]))
    #         print("\tENSEMBLE | EM: {:.4f} | F1: {:.4f}".format(a[0], a[1]))