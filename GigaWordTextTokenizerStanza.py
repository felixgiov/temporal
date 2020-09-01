import stanza
import gzip
from datetime import datetime

"""
Runtime:
    100,000 chars = 11 secs in moss
    1,000,000 chars = 5 mins 35 secs in moss
    10,000,000 chars = ? (More than 2 hours...) in moss
"""

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize')

filename = 'afp_eng_199405.txt.gz'
lines = ""
with gzip.open('/share/text/EnglishGigaword/text/afp_eng/'+filename, 'rt') as f:
    for line in f:
        lines += line.replace("\n", " ")

doc = nlp(lines[:100000])
print(datetime.now())

sentences = []
for i, sentence in enumerate(doc.sentences):
    sentences.append("#doc_afp_eng_199405_sent_"+str(i))
    for token in sentence.tokens:
        sentences.append(str(int(token.id)-1)+"\t"+token.text+"\tO\t_\t[\'N\']\t["+str(int(token.id)-1)+"]")

with open('/home/felix/projects/research/datasets/GigaWord/'+filename[:-7]+'_stanza.txt', 'w') as writer:
    for sen_line in sentences:
        writer.write(sen_line+"\n")
