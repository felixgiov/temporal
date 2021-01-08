# import re
# from allennlp.predictors.predictor import Predictor
#
# predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")
#
# # sentence = "He has outlasted and sometimes outsmarted eight American presidents. Fidel Castro invited John Paul to come for a reason."
# # sentence = "Telerate shares fell 50 cents on Friday to close at $20 each in New York Stock Exchange composite trading. Dow Jones shares also fell 50 cents to close at $36.125 in Big Board composite trading."
# sentence = 'Butinstead of providing the votes to strike it down , they chose to uphold it on the flimsy ground that because the sex of the parent and not the child made the difference under the law , the plaintiff did not have standing to bring the case.'
#
# srl = predictor.predict(sentence=sentence)
# print(srl)
#
# for verb in srl['verbs']:
#     if verb['verb'] == "fell":
#         args = re.findall(r'\[([^]]*)\]', verb['description'])
#         print(args)
#
# # {"verbs": [{"verb": "think", "description": "Did [ARG0: Uriah] [ARGM-ADV: honestly] [V: think] [ARG1: he could beat the game in under three hours] ?", "tags": ["O", "B-ARG0", "B-ARGM-ADV", "B-V", "B-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "O"]}, {"verb": "could", "description": "Did Uriah honestly think he [V: could] beat the game in under three hours ?", "tags": ["O", "O", "O", "O", "O", "B-V", "O", "O", "O", "O", "O", "O", "O", "O"]}, {"verb": "beat", "description": "Did Uriah honestly think [ARG0: he] [ARGM-MOD: could] [V: beat] [ARG1: the game] [ARGM-TMP: in under three hours] ?", "tags": ["O", "O", "O", "O", "B-ARG0", "B-ARGM-MOD", "B-V", "B-ARG1", "I-ARG1", "B-ARGM-TMP", "I-ARGM-TMP", "I-ARGM-TMP", "I-ARGM-TMP", "O"]}], "words": ["Did", "Uriah", "honestly", "think", "he", "could", "beat", "the", "game", "in", "under", "three", "hours", "?"]}
#
# # {'verbs': [
# # {'verb': 'outlasted', 'description': '[ARG0: He] has [V: outlasted] and sometimes outsmarted [ARG1: eight American presidents] . Fidel Castro invited John Paul to come for a reason .',
# # 'tags': ['B-ARG0', 'O', 'B-V', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']},
# # {'verb': 'outsmarted', 'description': '[ARG0: He] has outlasted and [ARGM-TMP: sometimes] [V: outsmarted] [ARG1: eight American presidents] . Fidel Castro invited John Paul to come for a reason .',
# # 'tags': ['B-ARG0', 'O', 'O', 'O', 'B-ARGM-TMP', 'B-V', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']},
# # {'verb': 'invited', 'description': 'He has outlasted and sometimes outsmarted eight American presidents . [ARG0: Fidel Castro] [V: invited] [ARG1: John Paul] [ARG2: to come] [ARGM-CAU: for a reason] .',
# # 'tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ARG0', 'I-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARG2', 'I-ARG2', 'B-ARGM-CAU', 'I-ARGM-CAU', 'I-ARGM-CAU', 'O']},
# # {'verb': 'come', 'description': 'He has outlasted and sometimes outsmarted eight American presidents . Fidel Castro invited [ARG1: John Paul] to [V: come] for a reason .',
# # 'tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1', 'O', 'B-V', 'O', 'O', 'O', 'O']}],
# # 'words': ['He', 'has', 'outlasted', 'and', 'sometimes', 'outsmarted', 'eight', 'American', 'presidents', '.', 'Fidel', 'Castro', 'invited', 'John', 'Paul', 'to', 'come', 'for', 'a', 'reason', '.']}
#
# # [('ABC19980120.1830.0957', '1', '1', ['People', 'have', 'predicted', 'his', 'demise', 'so', 'many', 'times', ',', 'and', 'the', 'US', 'has', 'tried', 'to', 'hasten', 'it', 'on', 'several', 'occasions', '.'], [2, 13], ['NNS', 'VBP', 'VBN', 'PRP$', 'NN', 'IN', 'JJ', 'NNS', ',', 'CC', 'DT', 'NNP', 'VBZ', 'VBN', 'TO', 'VB', 'PRP', 'IN', 'JJ', 'NNS', '.'], 'BEFORE'), ('ABC19980120.1830.0957', '1', '1', ['People', 'have', 'predicted', 'his', 'demise', 'so', 'many', 'times', ',', 'and', 'the', 'US', 'has', 'tried', 'to', 'hasten', 'it', 'on', 'several', 'occasions', '.'], [2, 15], ['NNS', 'VBP', 'VBN', 'PRP$', 'NN', 'IN', 'JJ', 'NNS', ',', 'CC', 'DT', 'NNP', 'VBZ', 'VBN', 'TO', 'VB', 'PRP', 'IN', 'JJ', 'NNS', '.'], 'BEFORE'), ('ABC19980120.1830.0957', '1', '1', ['People', 'have', 'predicted', 'his', 'demise', 'so', 'many', 'times', ',', 'and', 'the', 'US', 'has', 'tried', 'to', 'hasten', 'it', 'on', 'several', 'occasions', '.'], [13, 15], ['NNS', 'VBP', 'VBN', 'PRP$', 'NN', 'IN', 'JJ', 'NNS', ',', 'CC', 'DT', 'NNP', 'VBZ', 'VBN', 'TO', 'VB', 'PRP', 'IN', 'JJ', 'NNS', '.'], 'VAGUE'), ('ABC19980120.1830.0957', '1', '2', ['People', 'have', 'predicted', 'his', 'demise', 'so', 'many', 'times', ',', 'and', 'the', 'US', 'has', 'tried', 'to', 'hasten', 'it', 'on', 'several', 'occasions', '.', 'Time', 'and', 'again', ',', 'he', 'endures', '.'], [2, 26], ['NNS', 'VBP', 'VBN', 'PRP$', 'NN', 'IN', 'JJ', 'NNS', ',', 'CC', 'DT', 'NNP', 'VBZ', 'VBN', 'TO', 'VB', 'PRP', 'IN', 'JJ', 'NNS', '.', 'NNP', 'CC', 'RB', ',', 'PRP', 'VBZ', '.'], 'BEFORE'), ('ABC19980120.1830.0957', '1', '2', ['People', 'have', 'predicted', 'his', 'demise', 'so', 'many', 'times', ',', 'and', 'the', 'US', 'has', 'tried', 'to', 'hasten', 'it', 'on', 'several', 'occasions', '.', 'Time', 'and', 'again', ',', 'he', 'endures', '.'], [13, 26], ['NNS', 'VBP', 'VBN', 'PRP$', 'NN', 'IN', 'JJ', 'NNS', ',', 'CC', 'DT', 'NNP', 'VBZ', 'VBN', 'TO', 'VB', 'PRP', 'IN', 'JJ', 'NNS', '.', 'NNP', 'CC', 'RB', ',', 'PRP', 'VBZ', '.'], 'BEFORE'), ('ABC19980120.1830.0957', '1', '2', ['People', 'have', 'predicted', 'his', 'demise', 'so', 'many', 'times', ',', 'and', 'the', 'US', 'has', 'tried', 'to', 'hasten', 'it', 'on', 'several', 'occasions', '.', 'Time', 'and', 'again', ',', 'he', 'endures', '.'], [15, 26], ['NNS', 'VBP', 'VBN', 'PRP$', 'NN', 'IN', 'JJ', 'NNS', ',', 'CC', 'DT', 'NNP', 'VBZ', 'VBN', 'TO', 'VB', 'PRP', 'IN', 'JJ', 'NNS', '.', 'NNP', 'CC', 'RB', ',', 'PRP', 'VBZ', '.'], 'BEFORE')]
#
# # from difflib import SequenceMatcher
# #
# # print(SequenceMatcher(None, 'Dow Jones shares also fell 50 cents to close at $ 36.125 in Big Board composite trading', 'Telerate shares fell 50 cents on Friday').ratio())
# # print(SequenceMatcher(None, 'Telerate shares fell 50 cents on Friday to close at $ 20 each in New York Stock Exchange composite trading', 'Telerate shares fell 50 cents on Friday').ratio())
# # print(SequenceMatcher(None, 'Dow Jones shares also fell 50 cents to close at $ 36.125 in Big Board composite trading', 'Jones shares also fell 50 cents to close').ratio())
# # print(SequenceMatcher(None, 'Telerate shares fell 50 cents on Friday to close at $ 20 each in New York Stock Exchange composite trading', 'Jones shares also fell 50 cents to close').ratio())

from shapely.geometry import box
from shapely.ops import unary_union

p1 = box(2.2, 2.2, 2.5, 2.5)
# p2 = box(1.5, 1.5, 2.5, 2.5)
p2 = box(2.5, 2.5, 2.9, 2.9)
p3 = box(1, 1, 2, 2)


red_boxes = [p1, p2]
blue_boxes = [p3]
all_boxes = [p1, p2, p3]

red_union = unary_union(red_boxes)
blue_union = unary_union(blue_boxes)
all_union = unary_union(all_boxes)
resulting_intersection = red_union.intersection(blue_union)

print(red_union.area)
print(blue_union.area)
print(all_union.area)
print(resulting_intersection.area)
print(resulting_intersection.area/all_union.area)