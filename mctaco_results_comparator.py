single_task_mctaco = []
multi_task_mctaco = []
mctaco_gold = []
diff = []
diff2 = []
diff_mctaco_gold = []
mtl_same_to_mctaco = []
mtl_diff_to_mctaco = []
stl_same_to_mctaco = []
stl_diff_to_mctaco = []
same_to_mctaco = []
diff_to_mctaco = []

single_task_mctaco_dir = "/home/felix/projects/research/models/bert-base-uncased-og-383/pred_results_bert-base-uncased.txt"
multi_task_mctaco_dir = "/home/felix/projects/research/models/ner-from-tbaq-class-event-bert-base-uncased/pred_results_bert-base-uncased.txt"
mctaco_gold_dir = "/home/felix/projects/research/datasets/MCTACO/test_9442.tsv"

with open(single_task_mctaco_dir, "r") as reader:
    for row in reader:
        single_task_mctaco.append(row)

with open(multi_task_mctaco_dir, "r") as reader:
    for row in reader:
        multi_task_mctaco.append(row)

with open(mctaco_gold_dir, "r", encoding="utf-8") as reader:
    for row in reader:
        splits = row.split("\t")
        mctaco_gold.append((splits[0], splits[1], splits[2], splits[3], splits[4].replace("\n", "")))

event_ordering, event_duration, freq, stationarity, typical_time = 0, 0, 0, 0, 0
for row in mctaco_gold:
    if row[4] == "Event Ordering":
        event_ordering += 1
    elif row[4] == "Event Duration":
        event_duration += 1
    elif row[4] == "Frequency":
        freq += 1
    elif row[4] == "Stationarity":
        stationarity += 1
    elif row[4] == "Typical Time":
        typical_time += 1
print(event_ordering, event_duration, freq, stationarity, typical_time)

if len(single_task_mctaco) == len(multi_task_mctaco) == len(mctaco_gold):
    for i in range(len(single_task_mctaco)):
        if single_task_mctaco[i] != multi_task_mctaco[i] and multi_task_mctaco[i].replace("\n", "") == mctaco_gold[i][3]:
            diff.append((i,
                         single_task_mctaco[i].replace("\n", ""),
                         multi_task_mctaco[i].replace("\n", ""),
                         mctaco_gold[i][4],
                         mctaco_gold[i][0],
                         mctaco_gold[i][1],
                         mctaco_gold[i][2]))
        elif single_task_mctaco[i] != multi_task_mctaco[i] and multi_task_mctaco[i].replace("\n", "") != mctaco_gold[i][3]:
            diff2.append((i,
                         single_task_mctaco[i].replace("\n", ""),
                         multi_task_mctaco[i].replace("\n", ""),
                         mctaco_gold[i][4],
                         mctaco_gold[i][0],
                         mctaco_gold[i][1],
                         mctaco_gold[i][2],))

# print(diff)
# print(diff2)
print(len(diff))
print(len(diff2))
# print(diff_mctaco_gold)
# print(len(diff_mctaco_gold))

with open("became_correct_event.txt", "w") as writer:
    for row in diff:
        writer.write(str(row)+"\n")

with open("became_incorrect_event.txt", "w") as writer:
    for row in diff2:
        writer.write(str(row)+"\n")

stm_event_ordering, stm_event_duration, stm_freq, stm_stationarity, stm_typical_time = 0, 0, 0, 0, 0
for stm in diff:
    if stm[3] == "Event Ordering":
        stm_event_ordering += 1
    elif stm[3] == "Event Duration":
        stm_event_duration += 1
    elif stm[3] == "Frequency":
        stm_freq += 1
    elif stm[3] == "Stationarity":
        stm_stationarity += 1
    elif stm[3] == "Typical Time":
        stm_typical_time += 1
print(stm_event_ordering, stm_event_duration, stm_freq, stm_stationarity, stm_typical_time)

dtm_event_ordering, dtm_event_duration, dtm_freq, dtm_stationarity, dtm_typical_time = 0, 0, 0, 0, 0
for dtm in diff2:
    if dtm[3] == "Event Ordering":
        dtm_event_ordering += 1
    elif dtm[3] == "Event Duration":
        dtm_event_duration += 1
    elif dtm[3] == "Frequency":
        dtm_freq += 1
    elif dtm[3] == "Stationarity":
        dtm_stationarity += 1
    elif dtm[3] == "Typical Time":
        dtm_typical_time += 1
print(dtm_event_ordering, dtm_event_duration, dtm_freq, dtm_stationarity, dtm_typical_time)




if len(diff) == len(diff_mctaco_gold):
    for j in range(len(diff)):
        if diff[j][2] == mctaco_gold[j][3]:
            same_to_mctaco.append(diff_mctaco_gold[j])
        else:
            diff_to_mctaco.append(diff_mctaco_gold[j])

# print(same_to_mctaco)
print(len(same_to_mctaco))
# print(diff_to_mctaco)
print(len(diff_to_mctaco))

stm_event_ordering, stm_event_duration, stm_freq, stm_stationarity, stm_typical_time = 0, 0, 0, 0, 0
for stm in same_to_mctaco:
    if stm[4] == "Event Ordering":
        stm_event_ordering += 1
    elif stm[4] == "Event Duration":
        stm_event_duration += 1
    elif stm[4] == "Frequency":
        stm_freq += 1
    elif stm[4] == "Stationarity":
        stm_stationarity += 1
    elif stm[4] == "Typical Time":
        stm_typical_time += 1

if len(single_task_mctaco) == len(mctaco_gold):
    for j in range(len(single_task_mctaco)):
        if single_task_mctaco[j].replace("\n", "") == mctaco_gold[j][3]:
            stl_same_to_mctaco.append(mctaco_gold[j])
        else:
            stl_diff_to_mctaco.append(mctaco_gold[j])

print("Single Task")
print(len(stl_same_to_mctaco))
print(len(stl_diff_to_mctaco))

stl_stm_event_ordering, stl_stm_event_duration, stl_stm_freq, stl_stm_stationarity, stl_stm_typical_time = 0, 0, 0, 0, 0
for stl_stm in stl_same_to_mctaco:
    if stl_stm[4] == "Event Ordering":
        stl_stm_event_ordering += 1
    elif stl_stm[4] == "Event Duration":
        stl_stm_event_duration += 1
    elif stl_stm[4] == "Frequency":
        stl_stm_freq += 1
    elif stl_stm[4] == "Stationarity":
        stl_stm_stationarity += 1
    elif stl_stm[4] == "Typical Time":
        stl_stm_typical_time += 1

stl_dtm_event_ordering, stl_dtm_event_duration, stl_dtm_freq, stl_dtm_stationarity, stl_dtm_typical_time = 0, 0, 0, 0, 0
for stl_dtm in stl_diff_to_mctaco:
    if stl_dtm[4] == "Event Ordering":
        stl_dtm_event_ordering += 1
    elif stl_dtm[4] == "Event Duration":
        stl_dtm_event_duration += 1
    elif stl_dtm[4] == "Frequency":
        stl_dtm_freq += 1
    elif stl_dtm[4] == "Stationarity":
        stl_dtm_stationarity += 1
    elif stl_dtm[4] == "Typical Time":
        stl_dtm_typical_time += 1

print(stl_stm_event_ordering, stl_stm_event_duration, stl_stm_freq, stl_stm_stationarity, stl_stm_typical_time)
print(stl_dtm_event_ordering, stl_dtm_event_duration, stl_dtm_freq, stl_dtm_stationarity, stl_dtm_typical_time)

if len(multi_task_mctaco) == len(mctaco_gold):
    for j in range(len(multi_task_mctaco)):
        if multi_task_mctaco[j].replace("\n", "") == mctaco_gold[j][3]:
            mtl_same_to_mctaco.append(mctaco_gold[j])
        else:
            mtl_diff_to_mctaco.append(mctaco_gold[j])

print("Multi Tasks")
# print(same_to_mctaco)
print(len(mtl_same_to_mctaco))
# print(diff_to_mctaco)
print(len(mtl_diff_to_mctaco))

mtl_stm_event_ordering, mtl_stm_event_duration, mtl_stm_freq, mtl_stm_stationarity, mtl_stm_typical_time = 0, 0, 0, 0, 0
for mtl_stm in mtl_same_to_mctaco:
    if mtl_stm[4] == "Event Ordering":
        mtl_stm_event_ordering += 1
    elif mtl_stm[4] == "Event Duration":
        mtl_stm_event_duration += 1
    elif mtl_stm[4] == "Frequency":
        mtl_stm_freq += 1
    elif mtl_stm[4] == "Stationarity":
        mtl_stm_stationarity += 1
    elif mtl_stm[4] == "Typical Time":
        mtl_stm_typical_time += 1

mtl_dtm_event_ordering, mtl_dtm_event_duration, mtl_dtm_freq, mtl_dtm_stationarity, mtl_dtm_typical_time = 0, 0, 0, 0, 0
for mtl_dtm in mtl_diff_to_mctaco:
    if mtl_dtm[4] == "Event Ordering":
        mtl_dtm_event_ordering += 1
    elif mtl_dtm[4] == "Event Duration":
        mtl_dtm_event_duration += 1
    elif mtl_dtm[4] == "Frequency":
        mtl_dtm_freq += 1
    elif mtl_dtm[4] == "Stationarity":
        mtl_dtm_stationarity += 1
    elif mtl_dtm[4] == "Typical Time":
        mtl_dtm_typical_time += 1

print(mtl_stm_event_ordering, mtl_stm_event_duration, mtl_stm_freq, mtl_stm_stationarity, mtl_stm_typical_time)
print(mtl_dtm_event_ordering, mtl_dtm_event_duration, mtl_dtm_freq, mtl_dtm_stationarity, mtl_dtm_typical_time)

print("Diff to MCTACO, STL - MTL")
print(stl_dtm_event_ordering-mtl_dtm_event_ordering,
      stl_dtm_event_duration-mtl_dtm_event_duration,
      stl_dtm_freq-mtl_dtm_freq,
      stl_dtm_stationarity-mtl_dtm_stationarity,
      stl_dtm_typical_time-mtl_dtm_typical_time)

print("Same to MCTACO, STL - MTL")
print(stl_stm_event_ordering-mtl_stm_event_ordering,
      stl_stm_event_duration-mtl_stm_event_duration,
      stl_stm_freq-mtl_stm_freq,
      stl_stm_stationarity-mtl_stm_stationarity,
      stl_stm_typical_time-mtl_stm_typical_time)