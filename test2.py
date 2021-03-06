# import torch
#
# labels = torch.tensor([[-100, -100, -100, -100, 0, -100, 1, -100], [-100, 2, -100, -100, -100, 0, -100, -100]])
# print(labels.shape)
# print(labels)
# print(labels.view(-1).shape)
# print(labels.view(-1))
#
# active_labels = []
# print(active_labels)
# for lab in labels.view(-1):
#     if lab != -100:
#         active_labels.append(lab.item())
#
# active_labels = torch.tensor(active_labels).type_as(labels)
#
# print(active_labels.shape)
# print(active_labels)


# def segments_to_index_array(ner_segments):
#     per_batch_segments_index = []
#     for row in ner_segments:
#         per_sentence_segments_index = []
#         segments_sequence = []
#         for i, val in enumerate(row):
#             if val == -99 and (row[i-1] == -99 or row[i-1] == -98 or row[i-1] == -97 or row[i-1] == -96):
#                 if segments_sequence:
#                     per_sentence_segments_index.append(segments_sequence)
#                 segments_sequence = []
#                 segments_sequence.append(i)
#             elif val == -99 or val == -98 or val == -97 or val == -96:
#                     segments_sequence.append(i)
#             else:
#                 if segments_sequence:
#                     per_sentence_segments_index.append(segments_sequence)
#                 segments_sequence = []
#         per_batch_segments_index.append(per_sentence_segments_index)
#
#     return per_batch_segments_index
#
#
# a = [[ 0,   0,   0,   0, -99,   0,   0, -99,   0, -95, -93, -93, -99, -95,
#      -94, -93, -99,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#        0,   0]]
#
# b = segments_to_index_array(a)
#
# print(b)

# def normalize_numerals(inputString: str):
#     if hasNumbers(inputString) and not hasAlpha(inputString):
#         splitted_string = inputString.split(" ")
#         if len(splitted_string) > 1:
#             number = splitted_string[0]
#             fraction = splitted_string[1].split("/")
#             if int(number) >= 0:
#                 new_number = float(number) + (float(fraction[0]) / float(fraction[1]))
#             else:
#                 new_number = float(number) - (float(fraction[0]) / float(fraction[1]))
#         else:
#             number_or_fraction = splitted_string[0].split("/")
#             if len(number_or_fraction) > 1:
#                 new_number = float(number_or_fraction[0]) / float(number_or_fraction[1])
#             else:
#                 return inputString
#         return str(new_number)
#     else:
#         return inputString
#
# def hasNumbers(inputString):
#     return any(char.isdigit() for char in inputString)
#
# def hasAlpha(inputString):
#     return any(char.isalpha() for char in inputString)
#
# print(normalize_numerals("1/8"))
# print(normalize_numerals("2 5/8"))
# print(normalize_numerals("12"))
# print(normalize_numerals("-2 1/2"))
# print(normalize_numerals("AXXX"))
# print(normalize_numerals("200,000"))
# print(normalize_numerals("1.9"))
# print(normalize_numerals("100th"))
# print(normalize_numerals("7:15"))

# # for joint
# dict = {
#     "-1": "tmx0"
# }
#
# print(dict)
#
# dict_arr = []
# col_id_path = 'datasets/te3/te3-platinum_COL_ID/AP_20130322.tml'
# with open(col_id_path, "r") as reader:
#     for line_id in reader:
#         index = line_id.split("\t")[0]
#         id = line_id.split("\t")[1].replace('\n', '')
#         dict[index] = id
#         dict_arr.append(id)
#
# # for i, line_id in enumerate(dict):
# #     print(str(i)+" "+line_id)
#
# print(dict)
#
# for i, line_id in enumerate(dict_arr):
#     if line_id != "O":
#         if dict_arr[i+1] == dict_arr[i]:
#             dict_arr[i] = "O"
#             dict[str(i)] = "O"
#
# print(dict)
# # for i, line_id in enumerate(dict):
# #     print(str(i)+" "+line_id)
#
#
#
# def __getLinkStringRelEntID(self, links, entity_id, dict):
#     link_arr = []
#     link_str = ""
#     if entity_id in links:
#         for link in links[entity_id]:
#             link_str = self.__getEntityID(link[0])
#             link_str_converted = list(dict.keys())[list(dict.values()).index(link_str)]
#             link_arr.append(int(link_str_converted))
#             # link_str += self.__getEntityID(link[0]) + ","
#         # link_str = link_str[0:-1]
#     else:
#         link_str = "###"
#         link_arr.append(link_str)
#     return link_arr

#
# from transformers import AutoTokenizer
#
# bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
#
# inputs = bert_tokenizer.tokenize("[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]")
# print(inputs)
#
# inputs2 = roberta_tokenizer.tokenize("<s> Who was Jim Henson ? </s> Jim Henson was a puppeteer </s>")
# print(inputs2)

# from sklearn.model_selection import train_test_split

# from multi_utils import temprel_set
#
# temprel_trainset = temprel_set("datasets/MATRES/testset-temprel.xml")
#
# print(temprel_trainset.temprel_ee[0].token)
# print(temprel_trainset.temprel_ee[0].lemma)
# print(temprel_trainset.temprel_ee[0].part_of_speech)
# print(temprel_trainset.temprel_ee[0].position)
# print(temprel_trainset.temprel_ee[0].event_ix)
# print(temprel_trainset.temprel_ee[0].length)
# print(temprel_trainset.temprel_ee[0].label)

# temprel_train, temprel_dev = train_test_split(temprel_trainset.temprel_ee, test_size=0.2, random_state=2093)
#
# print(temprel_dev[0].token)

# with open('embeddings_sample.txt', 'r') as reader:
#     for line in reader:
#         a = line.split(" ")
#         print(len(a))

# import logging
# from transformers import RobertaTokenizer
# from multi_utils import MATRESProcessor, convert_matres_examples_to_features
#
# logger = logging.getLogger(__name__)
#
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
#
# a = tokenizer.tokenize("<s> Wearing </s>")
# print(a)
# a = tokenizer.convert_tokens_to_ids(a)
# print(a)
#
# # [0, 133, 1097, 1506, 438, 34298, 338, 17387,
# matres_processor = MATRESProcessor()
# matres_label_list = matres_processor.get_labels()
# matres_train_examples = matres_processor.get_train_examples('')
# matres_eval_examples = matres_processor.get_dev_examples('')
#
# train_dataset = convert_matres_examples_to_features(matres_train_examples,
#                                                            matres_label_list,
#                                                            128,
#                                                            tokenizer)
# eval_dataset = convert_matres_examples_to_features(matres_eval_examples,
#                                                           matres_label_list,
#                                                           128,
#                                                           tokenizer)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from multi_run import bigramGetter_fromNN

# embedding_dim = 1024
# lstm_hidden_dim = 64
# nn_hidden_dim = 64
# bigramStats_dim = 2
# bigramGetter = bigramGetter_fromNN
# output_dim = 4
# batch_size = 1
# granularity = 0.2
# common_sense_emb_dim = 3
# common_sense_emb = nn.Embedding(int(1.0 / granularity) * bigramStats_dim, common_sense_emb_dim)
#
# h_lstm2h_nn = nn.Linear(2 * lstm_hidden_dim + bigramStats_dim * common_sense_emb_dim,
#                              nn_hidden_dim)
# h_nn2o = nn.Linear(nn_hidden_dim + bigramStats_dim * common_sense_emb_dim, output_dim)
#
# e1_hidden_mean_batch = []
# e2_hidden_mean_batch = []
#
# output = torch.randn(2,16,128,1024)
# sequence_output = output[0]
# print(output.shape)
# print(sequence_output.shape)
# print(sequence_output[0][-99])
# for i in range(len(sequence_output)):
#     e1 = [-99,-99]
#     e2 = [-98]
#
#     list_e1 = []
#     list_e2 = []
#     for j in e1:
#         list_e1.append(sequence_output[i][j])
#     for k in e2:
#         list_e2.append(sequence_output[i][k])
#
#     e1_hidden_mean = torch.mean(torch.stack(list_e1), 0)
#     e2_hidden_mean = torch.mean(torch.stack(list_e2), 0)
#
#     e1_hidden_mean_batch.append(e1_hidden_mean)
#     e2_hidden_mean_batch.append(e2_hidden_mean)
#
#
# print(len(e1_hidden_mean_batch))
# print(len(e2_hidden_mean_batch))
#
# e1_hidden_mean_batch_stk = torch.stack(e1_hidden_mean_batch)
# e2_hidden_mean_batch_stk = torch.stack(e2_hidden_mean_batch)
#
# print(e1_hidden_mean_batch_stk.shape)
# print(e2_hidden_mean_batch_stk.shape)
#
# y = torch.cat((e1_hidden_mean_batch_stk, e2_hidden_mean_batch_stk), 1)
# print(y.shape)

# # common sense embeddings
# bigramstats = bigramGetter.getBigramStatsFromTemprel(temprel)
# common_sense_emb = common_sense_emb(torch.cuda.LongTensor(
#     [min(int(1.0 / granularity) - 1, int(bigramstats[0][0] / granularity))])).view(1, -1)
# for i in range(1, bigramStats_dim):
#     tmp = common_sense_emb(torch.cuda.LongTensor([(i - 1) * int(1.0 / granularity) + min(
#         int(1.0 / granularity) - 1, int(bigramstats[0][i] / granularity))])).view(1, -1)
#     common_sense_emb = torch.cat((common_sense_emb, tmp), 1)
#
# if not lowerCase:
#     embeds = emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
# else:
#     embeds = emb_cache.retrieveEmbeddings(tokList=[x.lower() for x in temprel.token]).cuda()
# embeds = embeds.view(temprel.length, batch_size, -1)
# lstm_out, hidden = lstm(embeds, hidden)
# lstm_out = lstm_out.view(embeds.size()[0], batch_size, lstm_hidden_dim)
# lstm_out = lstm_out[temprel.event_ix][:][:]
#
# h_nn = F.relu(h_lstm2h_nn(torch.cat((lstm_out.view(1, -1), common_sense_emb), 1)))
# output = h_nn2o(torch.cat((h_nn, common_sense_emb), 1))

# f = open("/home/felix/projects/research/datasets/MCTACO/dev_3783.tsv", "r")
# lines = [x.strip() for x in f.readlines()]
# mctaco_train, mctaco_dev = train_test_split(lines, test_size=0.1, random_state=2093)
#
# with open("/home/felix/projects/research/datasets/MCTACO/train_splitted.tsv", "w") as pred_writer:
#     for line in mctaco_train:
#         pred_writer.write(line+"\n")
#
# with open("/home/felix/projects/research/datasets/MCTACO/dev_splitted.tsv", "w") as pred_writer:
#     for line in mctaco_dev:
#         pred_writer.write(line+"\n")

# from transformers import RobertaTokenizer
#
# from multi_utils import TimeBankDurationProcessor, convert_tb_duration_examples_to_features
#
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
#
# tb_duration_processor = TimeBankDurationProcessor()
# tb_duration_label_list = tb_duration_processor.get_labels()
#
# tb_duration_train_examples = tb_duration_processor.get_train_examples('')
# tb_duration_train_dataset = convert_tb_duration_examples_to_features(tb_duration_train_examples,
#                                                                      tb_duration_label_list,
#                                                                      128,
#                                                                      tokenizer)
#
# tb_duration_all_examples = tb_duration_processor.get_all_examples('')
# tb_duration_all_dataset = convert_tb_duration_examples_to_features(tb_duration_all_examples,
#                                                                    tb_duration_label_list,
#                                                                    128,
#                                                                    tokenizer)
#
# print(len(tb_duration_train_dataset))
# print(len(tb_duration_all_dataset))

import sympy as sym

def calculateWeightedLossRatio(len_1, len_2, len_3, len_4, len_5):
    task_0_ratio = 0
    task_1_ratio = 0
    task_2_ratio = 0
    task_3_ratio = 0
    task_4_ratio = 0
    lengths = [len_1, len_2, len_3, len_4, len_5]
    active_lengths = {}
    act_lengths = []
    for i, length in enumerate(lengths):
        if length != 0:
            active_lengths[i] = length  # id start at 1
            act_lengths.append(length)

    n = len(active_lengths)

    # fix this to be more efficient
    if n==1:
        active_lengths[0] = 1
    elif n==2:
        a, b = sym.symbols('a,b')
        eq1 = sym.Eq(a + b, 1)
        eq2 = sym.Eq(act_lengths[0] * a, act_lengths[1] * b)
        result = sym.solve([eq1, eq2], (a, b))
        result_list = list(result.values())
        for i, key in enumerate(active_lengths):
            active_lengths[key] = result_list[i]
    elif n==3:
        a, b, c = sym.symbols('a,b,c')
        eq1 = sym.Eq(a + b + c, 1)
        eq2 = sym.Eq(act_lengths[0] * a, act_lengths[1] * b)
        eq3 = sym.Eq(act_lengths[1] * b, act_lengths[2] * c)
        result = sym.solve([eq1, eq2, eq3], (a, b, c))
        result_list = list(result.values())
        for i, key in enumerate(active_lengths):
            active_lengths[key] = result_list[i]
    elif n==4:
        a, b, c, d = sym.symbols('a,b,c,d')
        eq1 = sym.Eq(a + b + c + d, 1)
        eq2 = sym.Eq(act_lengths[0] * a, act_lengths[1] * b)
        eq3 = sym.Eq(act_lengths[1] * b, act_lengths[2] * c)
        eq4 = sym.Eq(act_lengths[2] * c, act_lengths[3] * d)
        result = sym.solve([eq1, eq2, eq3, eq4], (a, b, c, d))
        result_list = list(result.values())
        for i, key in enumerate(active_lengths):
            active_lengths[key] = result_list[i]
    elif n==5:
        a, b, c, d, e = sym.symbols('a,b,c,d,e')
        eq1 = sym.Eq(a + b + c + d + e, 1)
        eq2 = sym.Eq(act_lengths[0] * a, act_lengths[1] * b)
        eq3 = sym.Eq(act_lengths[1] * b, act_lengths[2] * c)
        eq4 = sym.Eq(act_lengths[2] * c, act_lengths[3] * d)
        eq5 = sym.Eq(act_lengths[3] * d, act_lengths[4] * e)
        result = sym.solve([eq1, eq2, eq3, eq4, eq5], (a, b, c, d, e))
        result_list = list(result.values())
        for i, key in enumerate(active_lengths):
            active_lengths[key] = result_list[i]
    else:
        active_lengths[0] = 0

    # dict = {}
    # for i in range(5):
    #     for key in active_lengths:
    #         if key == i:
    #             dict[i] = active_lengths[key]
    #         else:
    #             dict[i] = 0
    # print(dict)
    return active_lengths


calculateWeightedLossRatio(1000, 0, 0, 0, 0)
calculateWeightedLossRatio(1000, 0, 2000, 0, 0)
calculateWeightedLossRatio(1000, 0, 2000, 0, 3000)
calculateWeightedLossRatio(1000, 2000, 3000, 2500, 0)
calculateWeightedLossRatio(1000, 2000, 3000, 2500, 1000)
calculateWeightedLossRatio(0, 0, 0, 0, 0)


def construct_features_v2(dict_example_value):
    features_list = []
    context_dict = {}
    context = ""

    for item in dict_example_value:
        context_dict[item[1]+"_"+item[2]] = get_context(item[3])

    context = get_merged_context(context_dict)

    for i, item in enumerate(dict_example_value):

        features_positive_and_negative_example = []

        # context = get_context(item[3])
        event1_sentence = extract_event_sentence_spacy(item[3], item[4][0])
        event2_sentence = extract_event_sentence_spacy(item[3], item[4][1])

        question_after = 'What happened after ' + event1_sentence + '?'
        question_before = 'What happened before ' + event1_sentence + '?'

        if item[4] == "BEFORE":
            features_positive_and_negative_example.append(context + '\t' + question_after + '\t' + event2_sentence + '\tyes\tEvent Ordering')
            features_positive_and_negative_example.append(context + '\t' + question_before + '\t' + event2_sentence + '\tno\tEvent Ordering')
        elif item[4] == "AFTER":
            features_positive_and_negative_example.append(context + '\t' + question_before + '\t' + event2_sentence + '\tyes\tEvent Ordering')
            features_positive_and_negative_example.append(context + '\t' + question_after + '\t' + event2_sentence + '\tno\tEvent Ordering')
        else:
            features_positive_and_negative_example.append(context + '\t' + question_before + '\t' + event2_sentence + '\tno\tEvent Ordering')
            features_positive_and_negative_example.append(context + '\t' + question_after + '\t' + event2_sentence + '\tno\tEvent Ordering')

        features_list.append(features_positive_and_negative_example)

    return features_list

def get_merged_context(context_dict):
    context_sentence = ''
    context_list = []
    sorted_dict = sorted(context_dict.items())

    for item in sorted_dict:
        splitted_context = item[1].split(" .")
        for split_item in splitted_context:
            if split_item not in context_list:
                context_list.append(split_item)

    for context in context_list:
        context_sentence = context_sentence + context.strip() + ". "

    context_sentence.strip()

    return context_sentence

for key in dict_train_example:
    all_features.extend(construct_features_v2(dict_train_example[key]))

# df = pd.DataFrame(all_features).drop_duplicates()
# df.to_csv(path_or_buf='/home/felix/projects/research/datasets/MATRES-MCTACO/train12k_v2.tsv', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar="")

all_features = list(dict.fromkeys(all_features))

with open('/home/felix/projects/research/datasets/MATRES-MCTACO/train12k_v2.tsv', "w") as writer:
    for feature in all_features:
        writer.write(feature+'\n')