with open('/home/felix/projects/research/datasets/MATRES-MCTACO/train12k_v2.tsv', "r") as writer:
    for feature in writer:
        if feature.count("\t") != 4:
            print(feature)