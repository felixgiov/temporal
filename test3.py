# config, config_2, config_3, config_4, config_5, config_6, config_7, config_8, config_9, config_10, config_11, config_12, config_13, config_14, config_15, config_16, config_17, config_18, config_19, config_20, config_21, config_22, config_23, config_24, config_25, config_26, config_27, config_28, config_29, config_30,

from sklearn.model_selection import train_test_split

train, dev = None, None

def get_all_examples(self):
    f = open("/home/felix/projects/research/datasets/tb_duration/train_tb_duration.txt", "r")
    g = open("/home/felix/projects/research/datasets/tb_duration/test_tb_duration.txt", "r")
    lines = [x.strip() for x in f.readlines()]
    lines2 = [x.strip() for x in g.readlines()]
    lines.extend(lines2)
    return self._create_examples(lines, "train")

def split_examples(self):
    all_examples = self.get_all_examples()
    train, dev = train_test_split(all_examples, test_size=0.1, random_state=2093)
    self.train = train
    self.dev = dev

def get_train_examples(self, data_dir):
    # f = open("/home/felix/projects/research/datasets/tb_duration/train_tb_duration.txt", "r")
    # lines = [x.strip() for x in f.readlines()]
    # return self._create_examples(lines, "train")
    self.split_examples()
    return self.train

def get_test_examples(self, data_dir):
    # f = open("/home/felix/projects/research/datasets/tb_duration/test_tb_duration.txt", "r")
    # lines = [x.strip() for x in f.readlines()]
    # return self._create_examples(lines, "test")
    self.split_examples()
    return self.dev

def get_labels(self):
    return ["0", "1"]

def _create_examples(self, lines, type):
    examples = []
    for (i, line) in enumerate(lines):
        group = line.split("\t")
        guid = "%s-%s" % (type, i)
        text = group[0]
        index = group[1]
        label = group[4]
        examples.append(TBDurationInputExample(guid=guid, text=text, index=index, label=label))
    return examples