import json
import logging

from dataclasses import dataclass
import os
from sklearn.model_selection import train_test_split

from transformers import InputFeatures, DataProcessor, InputExample, PreTrainedTokenizer
from typing import List, Optional, Union
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    event_ids: Optional[List[int]] = None
    # cse_lemma : Optional[List[str]] = None
    # cse_position : Optional[List[str]] = None
    ner_token_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    label_ids: Optional[List[int]] = None
    task_name: Optional[int] = None


@dataclass(frozen=True)
class TimeBankInputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    ner_token_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    task_name: Optional[int] = None


@dataclass(frozen=True)
class TBDurationInputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    token_index: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    task_name: Optional[int] = None


@dataclass(frozen=True)
class MATRESInputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    event_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    task_name: Optional[int] = None


@dataclass(frozen=True)
class MCTACOInputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    task_name: Optional[int] = None


@dataclass(frozen=True)
class MATRESInputExample:
    guid: str
    text_a: List[str]
    event_ix: List[int]
    lemma: List[str]
    # position: List[str]
    label: Optional[str] = None


@dataclass(frozen=True)
class TimeBankInputExample:
    guid: str
    words: List[str]
    ner: Optional[List[str]]
    labels: Optional[List[str]]


@dataclass(frozen=True)
class TBDurationInputExample:
    guid: str
    text: List[str]
    index: Optional[int]
    label: Optional[str] = None


class TemporalProcessor(DataProcessor):

    # def get_train_examples(self, data_dir):
    #     f = open(os.path.join(data_dir, "dev_3783.tsv"), "r")
    #     lines = [x.strip() for x in f.readlines()]
    #     mctaco_train, mctaco_dev = train_test_split(lines, test_size=0.1, random_state=2093)
    #     return self._create_examples(mctaco_train, "train")
    #
    # def get_dev_examples(self, data_dir):
    #     f = open(os.path.join(data_dir, "dev_3783.tsv"), "r")
    #     lines = [x.strip() for x in f.readlines()]
    #     mctaco_train, mctaco_dev = train_test_split(lines, test_size=0.1, random_state=2093)
    #     return self._create_examples(mctaco_dev, "dev")

    def get_train_examples(self, data_dir):
        f = open("/home/felix/projects/research/datasets/MCTACO/train_splitted_sent.tsv", "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        f = open("/home/felix/projects/research/datasets/MCTACO/dev_splitted_sent.tsv", "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir):
        f = open("/home/felix/projects/research/datasets/MCTACO/test_9442.tsv", "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "test")

    def get_labels(self):
        return ["yes", "no"]

    def _create_examples(self, lines, type):
        examples = []
        for (i, line) in enumerate(lines):
            group = line.split("\t")
            guid = "%s-%s" % (type, i)
            text_a = group[0] + " " + group[1]
            text_b = group[2]
            label = group[3]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_mctaco_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for <s>, </s>, </s>, </s> with "- 4"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)
        else:
            # Account for <s> and </s> with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # RoBERTa
        tokens = []
        segment_ids = []
        tokens.append("<s>")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("</s>")
        segment_ids.append(0)
        tokens.append("</s>")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("</s>")
            segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        # features.append(
        #     InputFeatures(input_ids,
        #                   input_mask,
        #                   segment_ids,
        #                   segment_ids,  # event ids same with segment ids since it's all zero
        #                   # [],  # lemma for cse
        #                   # [],  # position for cse
        #                   [],  # ner token ids for timebank
        #                   label_id,
        #                   [],  # label ids for timebank
        #                   1))

        features.append(
            MCTACOInputFeatures(input_ids,
                                input_mask,
                                segment_ids,
                                label_id,
                                1))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


""" MATRES Part """


class temprel_ee:
    def __init__(self, xml_element):
        self.xml_element = xml_element
        self.label = xml_element.attrib['LABEL']
        self.sentdiff = int(xml_element.attrib['SENTDIFF'])
        self.docid = xml_element.attrib['DOCID']
        self.source = xml_element.attrib['SOURCE']
        self.target = xml_element.attrib['TARGET']
        self.data = xml_element.text.strip().split()
        self.token = []
        self.lemma = []
        self.part_of_speech = []
        self.position = []
        self.length = len(self.data)
        self.event_ix = []
        for i, d in enumerate(self.data):
            tmp = d.split('///')
            self.part_of_speech.append(tmp[-2])
            self.position.append(tmp[-1])
            if tmp[-1] == 'E1':
                self.event_ix.append(i)
            elif tmp[-1] == 'E2':
                self.event_ix.append(i)
            # self.token.append(d[:-(len(tmp[-1])+len(tmp[-2])+2)])
            self.token.append(tmp[0])
            self.lemma.append(tmp[1])


class temprel_set:
    def __init__(self, xmlfname, datasetname="matres"):
        self.xmlfname = xmlfname
        self.datasetname = datasetname
        tree = ET.parse(xmlfname)
        root = tree.getroot()
        self.size = len(root)
        self.temprel_ee = []
        for e in root:
            self.temprel_ee.append(temprel_ee(e))


class MATRESProcessor(DataProcessor):
    temprel_trainset = temprel_set("datasets/MATRES/trainset-temprel.xml")
    temprel_testset = temprel_set("datasets/MATRES/testset-temprel.xml")
    temprel_train, temprel_dev = train_test_split(temprel_trainset.temprel_ee, test_size=0.2, random_state=2093)

    # # uncomment if we want to predict test (put test into dev)
    # temprel_dev = temprel_testset.temprel_ee

    def get_all_train_examples(self, data_dir):
        return self._create_examples(self.temprel_trainset.temprel_ee, "train")

    def get_train_examples(self, data_dir):
        return self._create_examples(self.temprel_train, "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self.temprel_dev, "dev")

    def get_labels(self):
        return ["BEFORE", "AFTER", "EQUAL", "VAGUE"]

    def _create_examples(self, temprels, type):
        examples = []
        for (i, temprel) in enumerate(temprels):
            guid = "%s-%s" % (type, i)
            text_a = temprel.token
            event_ix = temprel.event_ix
            lemma = temprel.lemma
            # position = temprel.position
            label = temprel.label
            examples.append(MATRESInputExample(guid=guid,
                                               text_a=text_a,
                                               event_ix=event_ix,
                                               lemma=lemma,
                                               # position=position,
                                               label=label))
        return examples


def convert_matres_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        # add truncation
        tokens = []
        segment_ids = []
        event_ids = []
        tokens.append("<s>")
        segment_ids.append(0)
        event_ids.append(0)
        for i, tok in enumerate(example.text_a):
            if i == example.event_ix[0]:
                token = tokenizer.tokenize('<s> ' + example.lemma[i] + ' </s>')
                for tkn in token[1:-1]:
                    tokens.append(tkn)
                    segment_ids.append(0)
                    event_ids.append(-99)
                # token = tokenizer.tokenize('<s> '+tok+' </s>')
                # for tkn in token[1:-1]:
                #     tokens.append(tkn)
                #     segment_ids.append(0)
                #     event_ids.append(-99)
            elif i == example.event_ix[1]:
                token = tokenizer.tokenize('<s> ' + example.lemma[i] + ' </s>')
                for tkn in token[1:-1]:
                    tokens.append(tkn)
                    segment_ids.append(0)
                    event_ids.append(-98)
                # token = tokenizer.tokenize('<s> ' + tok + ' </s>')
                # for tkn in token[1:-1]:
                #     tokens.append(tkn)
                #     segment_ids.append(0)
                #     event_ids.append(-98)
            else:
                token = tokenizer.tokenize('<s> ' + tok + ' </s>')
                for tkn in token[1:-1]:
                    tokens.append(tkn)
                    segment_ids.append(0)
                    event_ids.append(0)
        tokens.append("</s>")
        segment_ids.append(0)
        event_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            event_ids.append(0)

        if len(input_ids) == max_seq_length:
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(event_ids) == max_seq_length

            label_id = label_map[example.label]
            if ex_index < 5:
                # print(tokens)
                # print(input_ids)
                # print(input_mask)
                # print(segment_ids)
                # print(event_ids)
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("event_ids: %s" % " ".join([str(x) for x in event_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label_id))

            # features.append(
            #     InputFeatures(input_ids,
            #                   input_mask,
            #                   segment_ids,
            #                   event_ids,
            #                   # example.lemma,  # for cse
            #                   # example.position,  # for cse
            #                   [],  # ner token ids for timebank
            #                   label_id,
            #                   [],  # label ids for timebank
            #                   4))

            features.append(
                MATRESInputFeatures(input_ids,
                                    input_mask,
                                    segment_ids,
                                    event_ids,
                                    label_id,
                                    4))
    return features


""" TimeBank Event and Timex Cls Part"""


class TimeBankProcessor(DataProcessor):

    def get_all_train_examples(self, data_dir):
        f = open("datasets/TBAQ-cleaned/timebank_v4.txt", "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "train")

    def get_train_examples(self, data_dir):
        f = open("datasets/TBAQ-cleaned/ner-from-tbaq/class-v3/train.txt", "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        f = open("datasets/TBAQ-cleaned/ner-from-tbaq/class-v3/dev.txt", "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def get_timex_labels(self):
        return ["DATE", "TIME", "DURATION", "SET"]

    def get_event_labels(self):
        return ["OCCURRENCE", "PERCEPTION", "REPORTING", "ASPECTUAL", "STATE", "I_STATE", "I_ACTION"]

    # def _create_examples(self, lines, type):
    #     examples = []
    #     for (i, line) in enumerate(lines):
    #         group = line.split("\t")
    #         guid = "%s-%s" % (type, i)
    #         text_a = group[0] + " " + group[1]
    #         text_b = group[2]
    #         label = group[3]
    #         examples.append(TimeBankInputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    #     return examples

    def _create_examples(self, lines, type):
        guid_index = 1
        examples = []
        words = []
        ner = []
        labels = []
        for line in lines:
            if line.startswith("#") or line == "" or line == "\n":
                if words:
                    examples.append(
                        TimeBankInputExample(guid=f"{type}-{guid_index}", words=words, ner=ner, labels=labels))
                    guid_index += 1
                    words = []
                    ner = []
                    labels = []
            else:
                splits = line.split("\t")
                words.append(splits[1])
                ner.append(splits[2])
                labels.append(splits[3])  # here
        if words:
            examples.append(TimeBankInputExample(guid=f"{type}-{guid_index}", words=words, ner=ner, labels=labels))
        return examples

    # def _create_examples(self, data_dir, mode: Union[Split, str]) -> List[TimeBankInputExample]:
    #     if isinstance(mode, Split):
    #         mode = mode.value
    #     file_path = os.path.join(data_dir, f"{mode}.txt")
    #     guid_index = 1
    #     examples = []
    #     with open(file_path, encoding="utf-8") as f:
    #         words = []
    #         ner = []
    #         labels = []
    #         for line in f:
    #             if line.startswith("#") or line == "" or line == "\n":
    #                 if words:
    #                     examples.append(TimeBankInputExample(guid=f"{mode}-{guid_index}", words=words, ner=ner, labels=labels))
    #                     guid_index += 1
    #                     words = []
    #                     ner = []
    #                     labels = []
    #             else:
    #                 splits = line.split("\t")
    #                 words.append(splits[1])
    #                 ner.append(splits[2])
    #                 labels.append(splits[3])  # here
    #         if words:
    #             examples.append(TimeBankInputExample(guid=f"{mode}-{guid_index}", words=words, ner=ner, labels=labels))
    #     return examples


def convert_timebank_examples_to_features(
        examples: List[TimeBankInputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        # data_dir: str,
        task_name: int,  # 2 for timex, 3 for event
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    label_map = {label: i for i, label in enumerate(label_list)}

    ner_segment_arr = []
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        ner_segment_ids = []
        labels = []

        if task_name == 2:
            labels = ["DATE", "TIME", "DURATION", "SET"]
        elif task_name == 3:
            labels = ["OCCURRENCE", "PERCEPTION", "REPORTING", "ASPECTUAL", "STATE", "I_STATE", "I_ACTION"]

        for word, label, ner in zip(example.words, example.labels, example.ner):
            word_tokens = tokenizer.tokenize('<s> ' + word + ' </s>')[1:-1]

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                # if label in ["OCCURRENCE", "PERCEPTION", "REPORTING", "ASPECTUAL", "STATE", "I_STATE", "I_ACTION"]:
                if label in labels:
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                else:
                    label_ids.extend([pad_token_label_id] * (len(word_tokens)))

                if ner == "B-EVENT":
                    ner_segment_ids.extend([-99] + [-98] * (len(word_tokens) - 1))
                elif ner == "I-EVENT":
                    ner_segment_ids.extend([-97] + [-96] * (len(word_tokens) - 1))
                elif ner == "B-TIMEX3":
                    ner_segment_ids.extend([-95] + [-94] * (len(word_tokens) - 1))
                elif ner == "I-TIMEX3":
                    ner_segment_ids.extend([-93] + [-92] * (len(word_tokens) - 1))
                else:
                    ner_segment_ids.extend([0] + [0] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            ner_segment_ids = ner_segment_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        ner_segment_ids += [0]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            ner_segment_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            ner_segment_ids += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            ner_segment_ids = [0] + ner_segment_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            ner_segment_ids = ([0] * padding_length) + ner_segment_ids
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            ner_segment_ids += [0] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(ner_segment_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("ner_segments: %s", " ".join([str(x) for x in ner_segment_ids]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        # features.append(
        #     InputFeatures(
        #         input_ids=input_ids,
        #         attention_mask=input_mask,
        #         token_type_ids=segment_ids,
        #         event_ids=[],
        #         ner_token_ids=ner_segment_ids,
        #         label=-1,  # label for mctaco and matres
        #         label_ids=label_ids,
        #         task_name=task_name,
        #     )
        # )

        features.append(
            TimeBankInputFeatures(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                ner_token_ids=ner_segment_ids,
                label_ids=label_ids,
                task_name=task_name,
            )
        )

    return features


"""
TimeBank Event Duration
"""


class TimeBankDurationProcessor(DataProcessor):
    def get_all_examples(self, data_dir):
        f = open("/home/felix/projects/research/datasets/tb_duration/train_tb_duration.txt", "r")
        g = open("/home/felix/projects/research/datasets/tb_duration/test_tb_duration.txt", "r")
        lines = [x.strip() for x in f.readlines()]
        lines2 = [x.strip() for x in g.readlines()]
        lines.extend(lines2)
        return self._create_examples(lines, "train")

    def get_train_examples(self, data_dir):
        f = open("/home/felix/projects/research/datasets/tb_duration/train_tb_duration.txt", "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "train")

    def get_test_examples(self, data_dir):
        f = open("/home/felix/projects/research/datasets/tb_duration/test_tb_duration.txt", "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "test")

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

def is_subtoken(word):
    if word[:2] == "##":
        return True
    else:
        return False

def convert_tb_duration_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens = example.text.split(" ")
        restored_text = []
        new_index = int(example.index)
        for i in range(len(tokens)):

            if i == int(example.index):
                new_index = len(restored_text)

            if not is_subtoken(tokens[i]) and (i + 1) < len(tokens) and is_subtoken(tokens[i + 1]):
                restored_text.append(tokens[i] + tokens[i + 1][2:])
                if (i + 2) < len(tokens) and is_subtoken(tokens[i + 2]):
                    restored_text[-1] = restored_text[-1] + tokens[i + 2][2:]
            elif not is_subtoken(tokens[i]):
                restored_text.append(tokens[i])

        tokens2 = []
        segment_ids = []
        token_index = []
        tokens2.append("<s>")
        segment_ids.append(0)
        token_index.append(0)
        for i, tok in enumerate(restored_text):
            token = tokenizer.tokenize('<s> ' + tok + ' </s>')
            for tkn in token[1:-1]:
                tokens2.append(tkn)
                segment_ids.append(0)
                if i == new_index:
                    token_index.append(1)
                else:
                    token_index.append(0)
        tokens2.append("</s>")
        segment_ids.append(0)
        token_index.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens2)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            token_index.append(0)

        if len(input_ids) == max_seq_length:
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(token_index) == max_seq_length

            label_id = label_map[example.label]
            if ex_index < 5:
                # print(tokens2)
                # print(input_ids)
                # print(input_mask)
                # print(segment_ids)
                # print(token_index)
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens2]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("token_index: %s" % " ".join([str(x) for x in token_index]))
                logger.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(
                TBDurationInputFeatures(input_ids,
                                        input_mask,
                                        segment_ids,
                                        token_index,
                                        label_id,
                                        5))
    return features
