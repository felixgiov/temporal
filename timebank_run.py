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
    GlueDataTrainingArguments)
from timebank_utils import NerDataset, Split, get_labels, extract_ner_segments, read_examples_from_file
from timebank_modeling_bert import BertForTokenEventClassification, temporalmultitask, BertForTemporalMultitask
# from timebank_modelling_roberta import RobertaForTemporalMultitask
from timebank_trainer import Trainer
from mctaco_temporal_utils import TemporalProcessor, convert_mctaco_examples_to_features

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

    mctaco_data_args = DataTrainingArguments(data_dir="./datasets/MCTACO",
                                             labels="./datasets/MCTACO/mctaco_labels.txt",
                                             max_seq_length=data_args.max_seq_length,
                                             overwrite_cache=data_args.overwrite_cache)

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

    # Prepare TimeBank task
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

    # dev_examples = read_examples_from_file(data_args.data_dir, Split.dev)
    # dev_ner_segments = extract_ner_segments(dev_examples, data_args.max_seq_length, tokenizer)

    # MCTACO Config, Tokenizer, Model
    mctaco_config = AutoConfig.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        num_labels=mctaco_num_labels,
        cache_dir=model_args.cache_dir,
        # finetuning_task=mctaco_data_args.task_name,
    )
    mctaco_tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    multitask_model = BertForTemporalMultitask.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=mctaco_config,
        cache_dir=model_args.cache_dir,
    )

    multitask_model.set_batch_size(training_args.per_gpu_train_batch_size)
    # multitask_model.set_ner_segments(train_ner_segments)

    # Get TimeBank datasets
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

    # Get MCTACO datasets
    mctaco_processor = TemporalProcessor()
    mctaco_label_list = mctaco_processor.get_labels()

    mctaco_train_examples = None
    mctaco_eval_examples = None

    if training_args.do_train:
        mctaco_train_examples = mctaco_processor.get_train_examples(mctaco_data_args.data_dir)

    if training_args.do_eval and training_args.local_rank == -1:
        mctaco_eval_examples = mctaco_processor.get_dev_examples(mctaco_data_args.data_dir)

    mctaco_train_dataset = convert_mctaco_examples_to_features(mctaco_train_examples,
                                                               mctaco_label_list,
                                                               mctaco_data_args.max_seq_length,
                                                               mctaco_tokenizer) if training_args.do_train else None
    mctaco_eval_dataset = convert_mctaco_examples_to_features(mctaco_eval_examples,
                                                              mctaco_label_list,
                                                              mctaco_data_args.max_seq_length,
                                                              mctaco_tokenizer) if training_args.do_eval else None

    def mctaco_compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return glue_compute_metrics("rte", preds, p.label_ids)

    # Initialize our Trainer
    trainer = Trainer(
        model=multitask_model,
        args=training_args,
        timebank_train_dataset=train_dataset,
        mctaco_train_dataset=mctaco_train_dataset,
        eval_dataset=mctaco_eval_dataset,
        compute_metrics=mctaco_compute_metrics
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

    # Evaluation
    results = {}
    if training_args.do_eval:
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
