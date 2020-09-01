import gzip
from datetime import datetime
from nltk.parse import CoreNLPParser

"""
cd to stanford-cor-nlp folder
then 
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
-status_port 9000 -port 9000 -timeout 15000 & 

Runtime:
    afp_eng_199405.txt (9,305,644 chars = 50 min in basil)
"""

print(str(datetime.now())+" Starting..")

filename = 'afp_eng_199405.txt.gz'
lines = ""
sent_text = []
with gzip.open('/share/text/EnglishGigaword/text/afp_eng/'+filename, 'rt') as f:
    for line in f:
        sent_text.append(line.replace("\n", ""))

parser = CoreNLPParser(url='http://localhost:9000')
sentences = []

for i, sentence in enumerate(sent_text):
    sentences.append("#doc_afp_eng_199405_sent_" + str(i))
    tokenized_text = parser.tokenize(sentence)
    for j, token in enumerate(tokenized_text):
        sentences.append(str(j) + "\t" + token + "\tO\t_\t[\'N\']\t[" + str(j) + "]")


with open('/home/felix/projects/research/datasets/GigaWord/'+filename[:-7]+'_corenlp.txt', 'w') as writer:
    for sen_line in sentences:
        writer.write(sen_line+"\n")

print(str(datetime.now())+" Finished!")