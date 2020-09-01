# there is still some problem with aquaint, aquaint.txt is 37530 lines, aquaint _v2.txt is 37529 lines.
# the problem is in line 13928

# another problem is token . . . in aquaint, it should be ... without spaces

import os

def normalize_numerals(inputString: str):
    if hasNumbers(inputString) and not hasAlpha(inputString):
        splitted_string = inputString.split(" ")
        if len(splitted_string) > 1:
            number = splitted_string[0]
            fraction = splitted_string[1].split("/")
            if int(number) >= 0:
                new_number = float(number) + (float(fraction[0]) / float(fraction[1]))
            else:
                new_number = float(number) - (float(fraction[0]) / float(fraction[1]))
        else:
            number_or_fraction = splitted_string[0].split("/")
            if len(number_or_fraction) > 1:
                new_number = float(number_or_fraction[0]) / float(number_or_fraction[1])
            else:
                return inputString
        return str(new_number)
    else:
        return inputString

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def hasAlpha(inputString):
    return any(char.isalpha() for char in inputString)


lines = []
# files_path = 'datasets/TBAQ-cleaned/TimeBank_joint/'
# files_path = 'datasets/TBAQ-cleaned/AQUAINT_joint/'
files_path = 'datasets/te3/te3-platinum_joint/'
for filename in os.listdir(files_path):
    prev_sentence_last_word_index = 0
    index = -1
    with open(os.path.join(files_path, filename), 'r') as f:
        for line in f:
            if line.startswith("#"):
                lines.append(line)
                prev_sentence_last_word_index = index+1
            else:
                items = line.split("\t")
                index = int(items[0])
                new_index = int(items[0]) - prev_sentence_last_word_index
                tokens = items[1]
                normalized_tokens = normalize_numerals(tokens)
                rel_head_id = items[5].strip("[").strip("]\n").split(", ")
                rel_head_id_int = []
                for id in rel_head_id:
                    if id == "-1":
                        id = int(id)
                    else:
                        id = int(id) - prev_sentence_last_word_index
                    rel_head_id_int.append(id)
                rel_head_id_str = ""
                for id_int in rel_head_id_int:
                    rel_head_id_str += str(id_int)+", "
                rel_head_id_str = rel_head_id_str[:-2]
                line = str(new_index)+"\t"+str(normalized_tokens)+"\t"+items[2]+"\t"+items[3]+"\t"+items[4]+"\t"+"["+rel_head_id_str+"]\n"
                lines.append(line)
    # lines.append("\n")

# # Max Index for every sentence. Need to fix manually remove the first and add last index.
# last_sentence_max_index = 0
# with open('datasets/TBAQ-cleaned/timebank_v2_max_index.txt', 'w') as w:
# # with open('datasets/TBAQ-cleaned/aquaint_v2_max_index.txt', 'w') as w:
# # with open('datasets/te3/te3-platinum_v2_max_index.txt', 'w') as w:
#     for line2 in lines:
#         if line2.startswith("#"):
#             w.write(str(last_sentence_max_index-1)+"\n")
#             last_sentence_max_index=0
#         else:
#             last_sentence_max_index+=1

# with open('datasets/TBAQ-cleaned/timebank_v2.txt', 'w') as w:
# with open('datasets/TBAQ-cleaned/aquaint_v2.txt', 'w') as w:
with open('datasets/te3/te3-platinum_v2.txt', 'w') as w:
    for line2 in lines:
        w.write(line2)

