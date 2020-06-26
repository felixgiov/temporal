# -*- coding: utf-8 -*-

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset, DataProcessor, InputExample, InputFeatures
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, BertForSequenceClassification, BertTokenizer
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
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
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

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
                              label_id))
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

def main():
    # Parameters
    model_args = ModelArguments(
        model_name_or_path="roberta-base",
    )
    data_args = DataTrainingArguments(task_name="rte", data_dir="./datasets/MCTACO")
    training_args = TrainingArguments(
        output_dir=f"./models/{model_args.model_name_or_path}",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_gpu_train_batch_size=32,
        per_gpu_eval_batch_size=128,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=500,
        logging_first_step=True,
        save_steps=1000,
        # evaluate_during_training=True,
    )

    set_seed(training_args.seed)
    num_labels = 2

    # Config, Tokenizer, Model
    config = RobertaConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        # finetuning_task=data_args.task_name,
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    model = RobertaForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    processor = TemporalProcessor()
    label_list = processor.get_labels()

    train_examples = None
    eval_examples = None

    if training_args.do_train:
        train_examples = processor.get_train_examples(data_args.data_dir)

    if training_args.do_eval and training_args.local_rank == -1:
        eval_examples = processor.get_dev_examples(data_args.data_dir)

    # Datasets
    train_dataset = convert_examples_to_features(train_examples, label_list, data_args.max_seq_length, tokenizer) if training_args.do_train else None
    eval_dataset = convert_examples_to_features(eval_examples, label_list, data_args.max_seq_length, tokenizer) if training_args.do_eval else None

    # Metrics
    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # # Evaluation
    # results = {}
    # if training_args.do_eval and training_args.local_rank in [-1, 0]:
    #     logger.info("*** Evaluate ***")
    #
    #     eval_datasets = [eval_dataset]
    #     for eval_dataset in eval_datasets:
    #         result = trainer.evaluate(eval_dataset=eval_dataset)
    #
    #         output_eval_file = os.path.join(
    #             training_args.output_dir, f"eval_results_{model_args.model_name_or_path}.txt"
    #         )
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results *****")
    #             for key, value in result.items():
    #                 logger.info("  %s = %s", key, value)
    #                 writer.write("%s = %s\n" % (key, value))
    #
    #         results.update(result)

    # Prediction
    if training_args.do_predict:
        preds = trainer.predict(eval_dataset)

        preds_tensor = torch.from_numpy(preds.predictions)
        preds_argmax = torch.argmax(preds_tensor, dim=1)

        # m = nn.Softmax(dim=1)
        # preds_softmax = m(preds_tensor)

        output_tensor_file = os.path.join(
          training_args.output_dir, f"pred_tensor_results_{model_args.model_name_or_path}.txt"
        )
        output_pred_file = os.path.join(
          training_args.output_dir, f"pred_results_{model_args.model_name_or_path}.txt"
        )
        with open(output_tensor_file, "w") as tensor_writer:
            for val in preds.predictions:
                tensor_writer.write(str(val)+"\n")

        with open(output_pred_file, "w") as pred_writer:
            for val in preds_argmax:
                if val == 0:
                    pred_writer.write("yes\n")
                else:
                    pred_writer.write("no\n")

            # for i in range(len(preds_tensor)):
            #     pred_writer.write(str(torch.argmax(preds_tensor))+"\n")

                # if torch.argmax(val) == 0:
                #     pred_writer.write("yes\n")
                # else:
                #     pred_writer.write("no\n")


if __name__ == '__main__':
    main()

