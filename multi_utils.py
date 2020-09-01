import json
import logging

from dataclasses import dataclass
import os
from sklearn.model_selection import train_test_split

from transformers import InputFeatures, DataProcessor, InputExample
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
    label: Optional[Union[int, float]] = None
    task_name : Optional[int] = None


@dataclass(frozen=True)
class MATRESInputExample:
    guid: str
    text_a: List[str]
    event_ix: List[int]
    lemma: List[str]
    # position: List[str]
    label: Optional[str] = None


class TemporalProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        f = open(os.path.join(data_dir, "dev_3783.tsv"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "test_9442.tsv"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

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

        features.append(
            InputFeatures(input_ids,
                          input_mask,
                          segment_ids,
                          segment_ids,  # event ids same with segment ids since it's all zero
                          # [],  # lemma
                          # [],  # position
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


""" TempRel Part """


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

    # put test into dev
    temprel_dev = temprel_testset.temprel_ee

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
                token = tokenizer.tokenize('<s> '+example.lemma[i]+' </s>')
                for tkn in token[1:-1]:
                    tokens.append(tkn)
                    segment_ids.append(0)
                    event_ids.append(-99)
            elif i == example.event_ix[1]:
                token = tokenizer.tokenize('<s> '+example.lemma[i]+' </s>')
                for tkn in token[1:-1]:
                    tokens.append(tkn)
                    segment_ids.append(0)
                    event_ids.append(-98)
            else:
                token = tokenizer.tokenize('<s> '+tok+' </s>')
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

            features.append(
                InputFeatures(input_ids,
                              input_mask,
                              segment_ids,
                              event_ids,
                              # example.lemma,
                              # example.position,
                              label_id,
                              4))
    return features
