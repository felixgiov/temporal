# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """


import logging
import os
import sys

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    # Trainer,
    TrainingArguments,
    set_seed,
    DataProcessor,
    InputExample,
    InputFeatures,
    AutoModelForSequenceClassification,
    glue_compute_metrics,
    GlueDataTrainingArguments as McTacoDataTrainingArguments)
from timebank_utils import NerDataset, Split, get_labels, extract_ner_segments, read_examples_from_file
from timebank_modeling_bert import BertForTokenEventClassification, temporalmultitask, BertForTemporalMultitask
from timebank_trainer import Trainer


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
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

# @dataclass
# class InputFeatures:
#     """
#     A single set of features of data.
#     Property names are the same names as the corresponding inputs to a model.
#     """
#
#     input_ids: List[int]
#     attention_mask: List[int]
#     token_type_ids: Optional[List[int]] = None
#     task_name: Optional[List[int]] = None
#     label_ids: Optional[List[int]] = None

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

        # task_name = [21]

        features.append(
                InputFeatures(input_ids,
                              input_mask,
                              segment_ids,
                              # task_name,
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
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # MCTACO Parameters
    mctaco_model_args = ModelArguments(
        model_name_or_path="bert-large-uncased",
    )
    mctaco_data_args = McTacoDataTrainingArguments(task_name="rte", data_dir="./datasets/MCTACO")
    mctaco_training_args = TrainingArguments(
        output_dir=f"./models/{model_args.model_name_or_path}",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_gpu_train_batch_size=16,
        per_gpu_eval_batch_size=64,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=500,
        logging_first_step=True,
        save_steps=1000,
        # evaluate_during_training=True,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    labels = get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Prepare MCTACO task
    mctaco_num_labels = 2

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    # train_examples = read_examples_from_file(data_args.data_dir, Split.train)
    # train_ner_segments = extract_ner_segments(train_examples, data_args.max_seq_length, tokenizer)
    #
    # dev_examples = read_examples_from_file(data_args.data_dir, Split.dev)
    # dev_ner_segments = extract_ner_segments(dev_examples, data_args.max_seq_length, tokenizer)

    model = BertForTokenEventClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # MCTACO Config, Tokenizer, Model
    mctaco_config = AutoConfig.from_pretrained(
        mctaco_model_args.model_name_or_path,
        num_labels=mctaco_num_labels,
        # finetuning_task=mctaco_data_args.task_name,
    )
    mctaco_tokenizer = AutoTokenizer.from_pretrained(
        mctaco_model_args.model_name_or_path,
    )
    mctaco_model = AutoModelForSequenceClassification.from_pretrained(
        mctaco_model_args.model_name_or_path,
        from_tf=bool(".ckpt" in mctaco_model_args.model_name_or_path),
        config=mctaco_config,
        cache_dir=mctaco_model_args.cache_dir,
    )

    multitask_timebank_model, multitask_mctaco_model = temporalmultitask(model_args, config, mctaco_model_args, mctaco_config)

    multitask_model = BertForTemporalMultitask.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # multitask_model.set_ner_segments(train_ner_segments)

    # Get datasets
    train_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    # Get MCTACO datasets
    mctaco_processor = TemporalProcessor()
    mctaco_label_list = mctaco_processor.get_labels()

    mctaco_train_examples = None
    mctaco_eval_examples = None

    if mctaco_training_args.do_train:
        mctaco_train_examples = mctaco_processor.get_train_examples(mctaco_data_args.data_dir)

    if mctaco_training_args.do_eval and mctaco_training_args.local_rank == -1:
        mctaco_eval_examples = mctaco_processor.get_dev_examples(mctaco_data_args.data_dir)

    mctaco_train_dataset = convert_mctaco_examples_to_features(mctaco_train_examples,
                                                               mctaco_label_list,
                                                               mctaco_data_args.max_seq_length,
                                                               mctaco_tokenizer) if mctaco_training_args.do_train else None
    mctaco_eval_dataset = convert_mctaco_examples_to_features(mctaco_eval_examples,
                                                              mctaco_label_list,
                                                              mctaco_data_args.max_seq_length,
                                                              mctaco_tokenizer) if mctaco_training_args.do_eval else None

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    def mctaco_compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return glue_compute_metrics(mctaco_data_args.task_name, preds, p.label_ids)

    # Initialize our Trainer
    trainer = Trainer(
        multitask_model=multitask_model,
        model=multitask_timebank_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=mctaco_eval_dataset, #here
        compute_metrics=mctaco_compute_metrics, #here
        mctaco_model=multitask_mctaco_model,
        mctaco_args=mctaco_training_args,
        mctaco_train_dataset=mctaco_train_dataset,
        mctaco_eval_dataset=mctaco_eval_dataset,
        mctaco_compute_metrics=mctaco_compute_metrics,
    )

    # Training
    if training_args.do_train and mctaco_training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and mctaco_training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        # test_dataset = NerDataset(
        #     data_dir=data_args.data_dir,
        #     tokenizer=tokenizer,
        #     labels=labels,
        #     model_type=config.model_type,
        #     max_seq_length=data_args.max_seq_length,
        #     overwrite_cache=data_args.overwrite_cache,
        #     mode=Split.test,
        # )
        #
        # predictions, label_ids, metrics = trainer.predict(test_dataset)
        # preds_list, _ = align_predictions(predictions, label_ids)
        #
        # output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        # if trainer.is_world_master():
        #     with open(output_test_results_file, "w") as writer:
        #         for key, value in metrics.items():
        #             logger.info("  %s = %s", key, value)
        #             writer.write("%s = %s\n" % (key, value))
        #
        # # Save predictions
        # output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        # if trainer.is_world_master():
        #     with open(output_test_predictions_file, "w") as writer:
        #         with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
        #             example_id = 0
        #             for line in f:
        #                 if line.startswith("-DOCSTART-") or line == "" or line == "\n":
        #                     writer.write(line)
        #                     if not preds_list[example_id]:
        #                         example_id += 1
        #                 elif preds_list[example_id]:
        #                     output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
        #                     writer.write(output_line)
        #                 else:
        #                     logger.warning(
        #                         "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0]
        #                     )
        preds = trainer.predict(mctaco_eval_dataset)

        preds_tensor = torch.from_numpy(preds.predictions)
        preds_argmax = torch.argmax(preds_tensor, dim=1)

        output_tensor_file = os.path.join(
            training_args.output_dir, f"pred_tensor_results_{model_args.model_name_or_path}.txt"
        )
        output_pred_file = os.path.join(
            training_args.output_dir, f"pred_results_{model_args.model_name_or_path}.txt"
        )
        with open(output_tensor_file, "w") as tensor_writer:
            for val in preds.predictions:
                tensor_writer.write(str(val) + "\n")

        with open(output_pred_file, "w") as pred_writer:
            for val in preds_argmax:
                if val == 0:
                    pred_writer.write("yes\n")
                else:
                    pred_writer.write("no\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()