# from nltk.parse.corenlp import CoreNLPParser
#
# jar = '/home/felix/tools/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar'
# parser = CoreNLPParser(jar)
#
# for i, sentence in enumerate(sent_text):
#     sentences.append("#doc_afp_eng_199405_sent_" + str(i))
#     tokenized_text = tk.tokenize(sentence)
#
#     for j, token in enumerate(tokenized_text):
#         sentences.append(str(j) + "\t" + token + "\tO\t_\t[\'N\']\t[" + str(j) + "]")
#
# with open('/home/felix/projects/research/datasets/GigaWord/'+filename[:-7]+'_nltkv3.txt', 'w') as writer:
#     for sen_line in sentences:
#         writer.write(sen_line+"\n")
#
# print(str(datetime.now())+" Finished!")

import spacy
from spacy.symbols import nsubj, VERB
nlp = spacy.load('en_core_web_sm')

text = 'Telerate shares fell 50 cents on Friday to close at $20 each in New York Stock Exchange composite trading . Dow Jones shares also fell 50 cents to close at $36.125 in Big Board composite trading .'
text_a = 'Telerate shares fell 50 cents on Friday to close'
text_b = 'Telerate shares fell'
text2 = 'Dow Jones shares also fell'
text_3 = 'In Delaware Chancery Court litigation , Telerate has criticized Dow Jones for not disclosing that Telerate \'s management expects the company \'s revenue to increase by 20 % annually , while Dow Jones based its projections of Telerate \'s performance on a 12 % revenue growth forecast . In the tender offer supplement , Dow Jones discloses the different growth forecasts but says it views the 20 % growth rate " as a hoped-for goal " of Telerate \'s management " and not as a realistic basis on which to project the company \'s likely future performance . '
text_4 = 'On Monday , Spitzer called for Vacco to revive that unit immediately , vowing that he would do so on his first day in office if elected .'
text_5 = 'He said the strength of the world-wide economy is suspect , and does n\'t see much revenue growth in the cards . He also said that the price wars flaring up in parts of the computer industry will continue through next year .'

doc = nlp(text_5)

for token in doc:
    print(token.text,'=>',token.dep_,'=>',token.head)
#
#
# doc=nlp(text)
#
# sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj") ]
#
# print(sub_toks)


# #  find subject
# subject = ('', 0)
# for i, token in enumerate(nlp(text)):
#     if token.dep_ == "nsubj":
#         subject = (token, i)
#
# #  find tokens that dependent on subject that appears before the subject
# index = subject[1]
# for token in nlp(text)[:index]:
#     if token.head == subject[0]:
#         print(token.text)


# doc = nlp("Autonomous cars shift insurance liability toward manufacturers")

# # Finding a verb with a subject from below — good
# verbs = set()
# for possible_subject in doc:
#     if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
#         verbs.add(possible_subject.head)
# print(verbs)

# print([token.text for token in doc[2].lefts])
# print([token.text for token in doc[2].rights])
# print(doc[2].n_lefts)
# print(doc[2].n_rights)

event = "close"
subj_chunk = []
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
    # # if chunk.root.dep_ == "nsubj" and chunk.root.head.text == event:
    # if chunk.root.dep_ == "nsubj":
    #     subj_chunk.append((chunk.text, chunk.end))

# for chunk in doc.noun_chunks:
#     if chunk.root.dep_ == "nsubj":
#         subj_chunk.append((chunk.text, chunk.end))

print(subj_chunk)

# print(subj_chunk.index(text))

# index = 13
# event_token = None
#
# if doc[index].text == "vowing":
#     event_token = doc[index]
#
# for parent in event_token.ancestors:
#     if parent.dep_ == "ROOT":
#         print(parent.text)

# while parent.dep_ != "ROOT":


# for token in doc:
#     print(token.text, token.dep_, token.head.text, token.head.pos_,
#             [child for child in token.children])