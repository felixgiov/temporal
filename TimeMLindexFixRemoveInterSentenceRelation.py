max_index = []
with open('datasets/TBAQ-cleaned/timebank_v2_max_index.txt', 'r') as reader:
# with open('datasets/TBAQ-cleaned/aquaint_v2_max_index.txt', 'r') as reader:
# with open('datasets/te3/te3-platinum_v2_max_index.txt', 'r') as reader:
    for line in reader:
        max_index.append(int(line))

new_lines = []
sentence_index = -1
with open('datasets/TBAQ-cleaned/timebank_v2.txt', 'r') as reader:
# with open('datasets/TBAQ-cleaned/aquaint_v2.txt', 'r') as reader:
# with open('datasets/te3/te3-platinum_v2.txt', 'r') as reader:
    for line in reader:
        if line.startswith("#"):
            new_lines.append(line)
            sentence_index += 1
        else:
            items = line.split("\t")
            index = int(items[0])
            relation = items[4].strip("[").strip("]").split(", ")
            rel_head_id = items[5].strip("[").strip("]\n").split(", ")
            rel = []
            rel_head_id_int = []
            for i, head_id in enumerate(rel_head_id):
                if not int(head_id) > max_index[sentence_index] and not int(head_id) < 0:
                    # v3 is without this if below.
                    if relation[i] in ["'BEFORE'", "'AFTER'", "'INCLUDES'", "'IS_INCLUDED'", "'DURING'",
                                       "'SIMULTANEOUS'", "'IAFTER'", "'IBEFORE'", "'IDENTITY'", "'BEGINS'",
                                       "'ENDS'", "'BEGUN_BY'", "'ENDED_BY'", "'DURING_INV'"]:
                        rel_head_id_int.append(head_id)
                        rel.append(relation[i])

            rel_str = ""
            if not rel:
                rel.append("'N'")
            for rel_item in rel:
                rel_str += rel_item + ", "
            rel_str = rel_str[:-2]

            rel_head_id_str = ""
            if not rel_head_id_int:
                rel_head_id_int.append(index)
            for id_int in rel_head_id_int:
                rel_head_id_str += str(id_int) + ", "
            rel_head_id_str = rel_head_id_str[:-2]

            new_line = items[0] + "\t" + items[1] + "\t" + items[2] + "\t" + items[3] + "\t" + "[" + rel_str + "]" + "\t" + "[" + rel_head_id_str + "]\n"
            new_lines.append(new_line)

with open('datasets/TBAQ-cleaned/timebank_v4.txt', 'w') as w:
# with open('datasets/TBAQ-cleaned/aquaint_v4.txt', 'w') as w:
# with open('datasets/te3/te3-platinum_v4.txt', 'w') as w:
    for line2 in new_lines:
        w.write(line2)

"""
Count number of intra-sentences relations
"""
# count = {}
# for line in new_lines:
#     if not line.startswith("#"):
#         items = line.split("\t")
#         relation = items[4].strip("[").strip("]").split(", ")
#         for rel in relation:
#             if rel not in count.keys():
#                 count[rel]=1
#             else:
#                 count[rel]+=1
#
# print(count)

