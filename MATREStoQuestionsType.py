from datetime import datetime
import xml.etree.ElementTree as ET
import pandas as pd
from itertools import groupby
from difflib import SequenceMatcher
import re
from sklearn.model_selection import train_test_split

from allennlp.predictors.predictor import Predictor

print(str(datetime.now())+" Starting..")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

# import spacy
# nlp = spacy.load('en_core_web_sm')

class temprel_ee:
    def __init__(self, xml_element):
        self.xml_element = xml_element
        self.label = xml_element.attrib['LABEL']
        self.sentdiff = int(xml_element.attrib['SENTDIFF'])
        self.docid = xml_element.attrib['DOCID']
        self.source = xml_element.attrib['SOURCE']
        self.target = xml_element.attrib['TARGET']
        self.source_sent_id = xml_element.attrib['SOURCE_SENTID']
        self.target_sent_id = xml_element.attrib['TARGET_SENTID']
        self.data = xml_element.text.strip().split()
        self.token = []
        self.lemma = []
        self.part_of_speech = []
        self.position = []
        self.length = len(self.data)
        self.event_ix = []
        for i, d in enumerate(self.data):
            tmp = d.split('///')
            self.part_of_speech.append(tmp[-2])
            self.position.append(tmp[-1])
            if tmp[-1] == 'E1':
                self.event_ix.append(i)
            elif tmp[-1] == 'E2':
                self.event_ix.append(i)
            # self.token.append(d[:-(len(tmp[-1])+len(tmp[-2])+2)])
            self.token.append(tmp[0])
            self.lemma.append(tmp[1])

class temprel_set:
    def __init__(self, xmlfname, datasetname="matres"):
        self.xmlfname = xmlfname
        self.datasetname = datasetname
        tree = ET.parse(xmlfname)
        root = tree.getroot()
        self.size = len(root)
        self.temprel_ee = []
        for e in root:
            self.temprel_ee.append(temprel_ee(e))

def create_examples(temprels):
    examples = []
    for (i, temprel) in enumerate(temprels):
        guid = temprel.docid+'_'+temprel.source_sent_id+'_'+temprel.target_sent_id
        text = temprel.token
        event_ix = temprel.event_ix
        pos_tag = temprel.part_of_speech
        # lemma = temprel.lemma
        # position = temprel.position
        label = temprel.label
        examples.append((guid, text, event_ix, pos_tag, label))
    return examples

def create_examples_v2(temprels):
    examples = []
    for (i, temprel) in enumerate(temprels):
        doc_id = temprel.docid
        source_id = temprel.source_sent_id
        target_id = temprel.target_sent_id
        text = temprel.token
        event_ix = temprel.event_ix
        pos_tag = temprel.part_of_speech
        # lemma = temprel.lemma
        # position = temprel.position
        label = temprel.label
        examples.append((doc_id, source_id, target_id, text, event_ix, pos_tag, label))
        # examples.append((doc_id, source_id, target_id, event_ix, label))
    return examples

def get_context(text):
    context_sentence = ''
    for word in text:
        context_sentence = context_sentence+word+' '

    context_sentence = context_sentence.strip()

    return context_sentence

def get_merged_context(context_dict):
    context_sentence = ''
    context_list = []
    sorted_dict = sorted(context_dict.items())

    for item in sorted_dict:

        i = (list(g) for _, g in groupby(item[1], key='.'.__ne__))
        xs = [a + b for a, b in zip(i, i)]

        for x in xs:
            if x not in context_list:
                context_list.append(x)

    # print(context_list)

    for item in context_list:
        context = get_context(item[:-1])
        context_sentence = context_sentence + context.strip() + ". "

    context_sentence.strip()

    return context_sentence

def getSVO(args):
    subject = ''
    verb = ''
    object = ''
    verbfound = False

    for item in args:
        splits = item.split(": ")
        # check len(splits) == 2 since some args don't have 'ARG0' or 'V', ex: 'of providing the votes to strike it down', 'ARG0: they', 'V: chose', 'ARG1: to uphold it on the ...'
        if len(splits) == 2:
            if splits[0] != 'V':
                if not verbfound:
                    subject = subject+splits[1]+" "
                else:
                    object = object+splits[1]+" "
            else:
                verb = verb+splits[1]
                verbfound = True

    subject = subject.strip()
    verb = verb.strip()
    object = object.strip()

    return (subject, verb, object)

def getTextAroundEvent(text, event_ix):

    textaroundevent = ''
    length = len(text)

    upper = event_ix
    bottom = event_ix

    if event_ix-3 < 0:
        bottom = 0
    else:
        bottom = event_ix-3

    # +6 since object generally longer and to account for the fullstop
    if event_ix+6 > length:
        upper = length
    else:
        upper = event_ix+6

    for i in range(bottom, upper):
        textaroundevent = textaroundevent+text[i]+' '

    textaroundevent = textaroundevent.strip()

    return textaroundevent


def extract_event_sentence(text, event_ix, event_1_or_2):
    sentence = ''

    if event_1_or_2 == 1:
        for word in text[:event_ix+1]:
            sentence = sentence+word+' '
    else:
        for word in text[event_ix:]:
            sentence = sentence+word+' '

    sentence = sentence.strip()

    return sentence

# def extract_event_sentence_spacy(text, event_ix, sent_id_1, sent_id_2):
#     subject = ''
#
#     txt = ' '.join(word for word in text)
#
#     word_num_diff_after_strip_split = 0
#     for i, word in enumerate(text):
#         if i < event_ix and '-' in word:
#             word_strip_count = 0
#             for char in word:
#                 if char == '-':
#                     word_strip_count += 2
#             word_num_diff_after_strip_split += word_strip_count
#
#     doc = nlp(txt)
#     subj_chunk = []
#
#     for chunk in doc.noun_chunks:
#         if chunk.root.dep_ == "nsubj":
#             subj_chunk.append((chunk.text, chunk.end))
#
#     # print(event_ix)
#     # print(subj_chunk)
#
#     for subj in subj_chunk:
#         if subj[1] <= event_ix+word_num_diff_after_strip_split:
#             subject = subj[0]
#
#     # event_obj = ''
#     # stopsymbols = ['.', ',']
#     #
#     # if sent_id_1 != sent_id_2:
#     #     for word in text[event_ix+1:]:
#     #         if word not in stopsymbols:
#     #             event_obj = event_obj+word+' '
#     #         else:
#     #             break
#     #
#     # sentence = subject+' '+text[event_ix]+' '+event_obj
#     sentence = subject + ' ' + text[event_ix]
#     sentence = sentence.strip()
#
#     return sentence

def extract_event_sentence_srl_bert(text, event_ix, srl, isQuestion):
    subject = ''

    # txt = ' '.join(word for word in text)
    # srl = predictor.predict(sentence=txt)

    srl_results = []
    svo_tuple = ()
    text_around_event = getTextAroundEvent(text, event_ix)

    for verb in srl['verbs']:
        if verb['verb'] == text[event_ix]:
            args = re.findall(r'\[([^]]*)\]', verb['description'])
            srl_results.append(getSVO(args))

    if len(srl_results) == 1:
        svo_tuple = srl_results[0]
    else:
        highest_ratio = 0.0
        for tuple in srl_results:
            ratio = SequenceMatcher(None, tuple[0] + ' ' + tuple[1] + ' ' + tuple[2], text_around_event).ratio()
            if ratio > highest_ratio:
                svo_tuple = tuple
                highest_ratio = ratio

    if svo_tuple:
        if isQuestion:
            sentence = svo_tuple[0] + ' ' + svo_tuple[1]
            sentence = sentence.strip()
        else:
            sentence = svo_tuple[0] + ' ' + svo_tuple[1] + ' ' + svo_tuple[2]
            sentence = sentence.strip()

        return sentence
    else:
        return "NO_EVENT_IN_SRL"


# def construct_features(dict_example_value):
#     features_list = []
#     context = ''
#     question = ''
#     candidate_answers = []
#     label = ''
#
#     for i, item in enumerate(dict_example_value):
#
#         features_positive_and_negative_example = []
#
#         context = get_context(item[1])
#         event1_sentence = extract_event_sentence_spacy(item[1], item[2][0])
#         event2_sentence = extract_event_sentence_spacy(item[1], item[2][1])
#
#         if item[4] == "BEFORE":
#             question = 'What happened after '+event1_sentence+'?'
#         elif item[4] == "AFTER":
#             question = 'What happened before '+event1_sentence+'?'
#         else:
#             print('VAGUE or EQUAL')
#             continue
#
#         # features_positive_and_negative_example.append(question + '\t' + event2_sentence + '\tyes\tEvent Ordering')
#         features_positive_and_negative_example.append(context + '\t' + question+'\t'+event2_sentence+'\tyes\tEvent Ordering')
#
#         # negative example part
#         negative_events_ix = set()
#         for j, item2 in enumerate(dict_example_value):
#
#             # if item2 is not item and item2's label are different than items label and first and second events are different
#             if j != i:
#                 if item[2][0] == item2[2][0] and item[2][1] != item2[2][1] and item2[4] == item[4]:
#                     continue
#                 elif item[2][0] == item2[2][0] and item[2][1] != item2[2][1] and item2[4] != item[4]:
#                     negative_events_ix.add(item2[2][1])
#                 elif item[2][0] != item2[2][0] and item[2][1] == item2[2][1] and item2[4] == item[4]:
#                     negative_events_ix.add(item2[2][0])
#                 elif item[2][0] != item2[2][0] and item[2][1] == item2[2][1] and item2[4] != item[4]:
#                     negative_events_ix.add(item2[2][0])
#                 elif item[2][0] != item2[2][0] and item[2][1] != item2[2][1] and item2[4] == item[4]:
#                     negative_events_ix.add(item2[2][0])
#                     negative_events_ix.add(item2[2][1])
#                 elif item[2][0] != item2[2][0] and item[2][1] != item2[2][1] and item2[4] != item[4]:
#                     negative_events_ix.add(item2[2][0])
#                     negative_events_ix.add(item2[2][1])
#
#         for negative_ix in negative_events_ix:
#             event2_sentence = extract_event_sentence_spacy(item[1], negative_ix)
#             # features_positive_and_negative_example.append(question + '\t' + event2_sentence + '\tno\tEvent Ordering')
#             features_positive_and_negative_example.append(context + '\t' + question+'\t'+event2_sentence+'\tno\tEvent Ordering')
#
#         features_list.append(features_positive_and_negative_example)
#
#     return features_list
#
# # construct feature : list of list of features, construct feature 2 : list of features
#
# def construct_features_(dict_example_value):
#     features_list = []
#     context = ''
#     question = ''
#     candidate_answers = []
#     label = ''
#
#     for i, item in enumerate(dict_example_value):
#
#         context = get_context(item[1])
#         event1_sentence = extract_event_sentence_spacy(item[1], item[2][0])
#         event2_sentence = extract_event_sentence_spacy(item[1], item[2][1])
#
#         if item[6] == "BEFORE":
#             question = 'What happened after '+event1_sentence+'?'
#             features_list.append(question + '\t' + event2_sentence + '\tyes\tEvent Ordering')
#             # features_list.append(context + '\t' + question+'\t'+event2_sentence+'\tyes\tEvent Ordering')
#         elif item[6] == "AFTER":
#             question = 'What happened after '+event1_sentence+'?'
#             features_list.append(question + '\t' + event2_sentence + '\tno\tEvent Ordering')
#             # features_list.append(context + '\t' + question+'\t'+event2_sentence+'\tno\tEvent Ordering')
#         else:
#             print('VAGUE or EQUAL')
#             continue
#
#     return features_list
#
#
# def construct_features_v2(dict_example_value):
#     features_list = []
#     context_dict = {}
#     context = ""
#
#     for item in dict_example_value:
#         context_dict[item[1]+"_"+item[2]] = item[3]
#
#     context = get_merged_context(context_dict).strip()
#
#     for i, item in enumerate(dict_example_value):
#
#         # context = get_context(item[3])
#         event1_sentence = extract_event_sentence_spacy(item[3], item[4][0], item[1], item[2])
#         event2_sentence = extract_event_sentence_spacy(item[3], item[4][1], item[1], item[2])
#
#         question_after = 'What happened after ' + event1_sentence + '?'
#         question_before = 'What happened before ' + event1_sentence + '?'
#
#         if item[6] == "BEFORE":
#             features_list.append(context + '\t' + question_after + '\t' + event2_sentence + '\tyes\tEvent Ordering')
#             features_list.append(context + '\t' + question_before + '\t' + event2_sentence + '\tno\tEvent Ordering')
#         elif item[6] == "AFTER":
#             features_list.append(context + '\t' + question_before + '\t' + event2_sentence + '\tyes\tEvent Ordering')
#             features_list.append(context + '\t' + question_after + '\t' + event2_sentence + '\tno\tEvent Ordering')
#         else:
#             features_list.append(context + '\t' + question_before + '\t' + event2_sentence + '\tno\tEvent Ordering')
#             features_list.append(context + '\t' + question_after + '\t' + event2_sentence + '\tno\tEvent Ordering')
#
#     return features_list

def construct_features_srl(dict_example_value):
    features_list = []
    context_dict = {}
    context = ""

    for item in dict_example_value:
        context_dict[item[1]+"_"+item[2]] = item[3]

    context = get_merged_context(context_dict).strip()

    # txt = ' '.join(word for word in dict_example_value[0][3])
    srl = predictor.predict(sentence=context)

    # reversed_dict_example_value = []
    # for item in dict_example_value:
    #     new_label = ''
    #     new_event_ix = [item[4][1], item[4][0]]
    #     if item[6] == 'BEFORE':
    #         new_label = 'AFTER'
    #     elif item[6] == 'AFTER':
    #         new_label = 'BEFORE'
    #     elif item[6] == 'EQUAL':
    #         new_label = 'EQUAL'
    #     else:
    #         new_label = 'VAGUE'
    #
    #     tuple = (item[0], item[1], item[2], item[3], new_event_ix, item[5], new_label)
    #     reversed_dict_example_value.append(tuple)
    #
    # dict_example_value.extend(reversed_dict_example_value)

    for i, item in enumerate(dict_example_value):

        # context = get_context(item[3])
        event1_sentence = extract_event_sentence_srl_bert(item[3], item[4][0], srl, True)
        event2_sentence = extract_event_sentence_srl_bert(item[3], item[4][1], srl, False)

        if event1_sentence == "NO_EVENT_IN_SRL" or event2_sentence == "NO_EVENT_IN_SRL":
            continue

        question_after = 'What happened after ' + event1_sentence + '?'
        question_before = 'What happened before ' + event1_sentence + '?'

        if item[6] == "BEFORE":
            features_list.append(context + '\t' + question_after + '\t' + event2_sentence + '\tyes\tEvent Ordering')
            features_list.append(context + '\t' + question_before + '\t' + event2_sentence + '\tno\tEvent Ordering')
        elif item[6] == "AFTER":
            features_list.append(context + '\t' + question_before + '\t' + event2_sentence + '\tyes\tEvent Ordering')
            features_list.append(context + '\t' + question_after + '\t' + event2_sentence + '\tno\tEvent Ordering')
        else:
            features_list.append(context + '\t' + question_before + '\t' + event2_sentence + '\tno\tEvent Ordering')
            features_list.append(context + '\t' + question_after + '\t' + event2_sentence + '\tno\tEvent Ordering')

    return features_list

temprel_trainset = temprel_set("datasets/MATRES/trainset-temprel.xml")
# temprel_trainset = temprel_set("datasets/MATRES/trainset_sample.xml")
temprel_testset = temprel_set("datasets/MATRES/testset-temprel.xml")
temprel_train, temprel_dev = train_test_split(temprel_trainset.temprel_ee, test_size=0.2, random_state=2093)

# # ================================================ V1 ============================================================
# train_example = create_examples(temprel_trainset.temprel_ee)
# dict_train_example = {}
#
# for tuple_row in train_example:
#     if tuple_row[0] not in dict_train_example:
#         dict_train_example[tuple_row[0]] = []
#         dict_train_example[tuple_row[0]].append(tuple_row)
#     else:
#         dict_train_example[tuple_row[0]].append(tuple_row)
#
# # print(dict_train_example)
# # print(len(dict_train_example))
# # print(len(dict_train_example['wsj_0542_7_8']))
# # print(dict_train_example['wsj_0542_7_8'])
# # print(construct_features(dict_train_example['wsj_0542_7_8']))
# # print(len(dict_train_example['wsj_0542_5_6']))
# # print(dict_train_example['wsj_0542_5_6'])
# # print(construct_features(dict_train_example['NYT19981026.0446_29_29']))
#
# # a = construct_features(dict_train_example['wsj_0542_7_8'])
# # b = construct_features(dict_train_example['wsj_0586_4_4'])
# # a.extend(b)
# #
# # for s in a:
# #     for t in s:
# #         print(t)
# #     print("======")
#
# # # for s in a:
# # #     print(s)
#
# with open('/home/felix/projects/research/datasets/MATRES-MCTACO/train12k.tsv', "w") as writer:
#     for key in dict_train_example:
#         features = construct_features(dict_train_example[key])
#         for feature in features:
#             for item in feature:
#                 writer.write(item+'\n')
# # ================================================ V1 ============================================================

# ================================================ V2 ============================================================
train_example = create_examples_v2(temprel_train.temprel_ee)
dict_train_example = {}
all_features = []

# 3 sentences
# for tuple_row in train_example:
#     if tuple_row[0]+"_"+tuple_row[1] not in dict_train_example:
#         dict_train_example[tuple_row[0] + "_" + tuple_row[1]] = []
#     elif tuple_row[0]+"_"+tuple_row[2] not in dict_train_example:
#         dict_train_example[tuple_row[0] + "_" + tuple_row[2]] = []
#
# for key in dict_train_example:
#     for tuple_row in train_example:
#         if tuple_row[0]+"_"+tuple_row[1] == key or tuple_row[0]+"_"+tuple_row[2] == key:
#             dict_train_example[key].append(tuple_row)

for tuple_row in train_example:
    if tuple_row[0]+"_"+tuple_row[1] not in dict_train_example:
        dict_train_example[tuple_row[0] + "_" + tuple_row[1]] = []

for key in dict_train_example:
    for tuple_row in train_example:
        if tuple_row[0]+"_"+tuple_row[1] == key:
            dict_train_example[key].append(tuple_row)


# print(dict_train_example['wsj_0542_7'])
# print(len(dict_train_example['wsj_0542_7']))
# print(len(dict_train_example))

# all_features.extend(construct_features_srl(dict_train_example['wsj_0542_7']))

for key in dict_train_example:
    all_features.extend(construct_features_srl(dict_train_example[key]))

# df = pd.DataFrame(all_features).drop_duplicates()
# df.to_csv(path_or_buf='/home/felix/projects/research/datasets/MATRES-MCTACO/train12k_v2.tsv', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar="")

# all_features = list(dict.fromkeys(all_features))

with open('/home/felix/projects/research/datasets/MATRES-MCTACO/train_srl_small.tsv', "w") as writer:
    for feature in all_features:
        writer.write(feature+'\n')

print(str(datetime.now())+" Finished!")