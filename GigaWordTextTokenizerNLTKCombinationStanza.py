import gzip
from datetime import datetime
import nltk
import stanza
import os

"""
Runtime:
    afp_eng_199405.txt (9,305,644 chars = 5 mins in moss)
"""

print(str(datetime.now())+" Starting..")

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize')

old_dir = "/share/text/EnglishGigaword/text/afp_eng/"
dir = "/pear/_backup/text/EnglishGigaword/text/afp_eng/"
filename = 'afp_eng_199405.txt.gz'

for filename in os.listdir(dir):
    sent_text = []
    with gzip.open(dir+filename, 'rt') as f:
        for line in f:
            sent_text.append(line.replace("\n", ""))

    sentences = []

    for i, sentence in enumerate(sent_text):
        sentences.append("#doc_afp_eng_199405_sent_" + str(i))

        if "-" in sentence:
            doc = nlp(sentence)
            tokens_arr = []
            for sen in doc.sentences:
                for token in sen.tokens:
                    tokens_arr.append(token.text)

            for k, tok in enumerate(tokens_arr):
                sentences.append(str(k) + "\t" + tok + "\tO\t_\t[\'N\']\t[" + str(k) + "]")


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

    with open('/home/felix/projects/research/datasets/GigaWord/afp_eng' + filename[:-7] + '.txt', 'w') as writer:
        for sen_line in sentences:
            writer.write(sen_line+"\n")

    print(str(datetime.now()) + " " + filename + " Done!")

print(str(datetime.now())+" Finished!")