import gzip
from datetime import datetime
import nltk
from nltk.tokenize.stanford import StanfordTokenizer

"""
Runtime:
    afp_eng_199405.txt (9,305,644 chars = 16 secs in moss, 1 min in basil)
"""

print(str(datetime.now())+" Starting..")

old_dir = "/share/text/EnglishGigaword/text/afp_eng/"
dir = "/pear/_backup/text/EnglishGigaword/text/afp_eng/"
filename = 'afp_eng_199405.txt.gz'
lines = ""
sent_text = []
with gzip.open(dir+filename, 'rt') as f:
    for line in f:
        sent_text.append(line.replace("\n", ""))

sentences = []

jar = '/home/felix/tools/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar'
tk = StanfordTokenizer(jar)

for i, sentence in enumerate(sent_text):
    sentences.append("#doc_afp_eng_199405_sent_" + str(i))

    if "-" in sentence:
        tokenized_text = tk.tokenize(sentence)

        for j, token in enumerate(tokenized_text):
            sentences.append(str(j) + "\t" + token + "\tO\t_\t[\'N\']\t[" + str(j) + "]")

    else:
        tokenized_text = nltk.word_tokenize(sentence)
        tokenized_text_fixed = []

        for text in tokenized_text:
            if text == "``" or text == "''":
                tokenized_text_fixed.append('"')
            else:
                tokenized_text_fixed.append(text)

        for j, token in enumerate(tokenized_text_fixed):
            sentences.append(str(j) + "\t" + token + "\tO\t_\t[\'N\']\t[" + str(j) + "]")

with open('/home/felix/projects/research/datasets/GigaWord/'+filename[:-7]+'_nltkv4.txt', 'w') as writer:
    for sen_line in sentences:
        writer.write(sen_line+"\n")

print(str(datetime.now())+" Finished!")