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


def segments_to_index_array(ner_segments):
    per_batch_segments_index = []
    for row in ner_segments:
        per_sentence_segments_index = []
        segments_sequence = []
        for i, val in enumerate(row):
            if val == -99 and (row[i-1] == -99 or row[i-1] == -98 or row[i-1] == -97 or row[i-1] == -96):
                if segments_sequence:
                    per_sentence_segments_index.append(segments_sequence)
                segments_sequence = []
                segments_sequence.append(i)
            elif val == -99 or val == -98 or val == -97 or val == -96:
                    segments_sequence.append(i)
            else:
                if segments_sequence:
                    per_sentence_segments_index.append(segments_sequence)
                segments_sequence = []
        per_batch_segments_index.append(per_sentence_segments_index)

    return per_batch_segments_index


a = [[ 0,   0,   0,   0, -99,   0,   0, -99,   0, -95, -93, -93, -99, -95,
     -94, -93, -99,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0]]

b = segments_to_index_array(a)

print(b)