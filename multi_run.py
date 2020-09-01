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

from transformers import (
    AutoConfig,
    RobertaConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    RobertaTokenizer,
    EvalPrediction,
    HfArgumentParser,
    # Trainer,
    TrainingArguments,
    # set_seed,
    DataProcessor,
    InputExample,
    InputFeatures,
    AutoModelForSequenceClassification,
    glue_compute_metrics,
    GlueDataTrainingArguments)

from multi_modelling_roberta import RobertaForTemporalMulti, VerbNet, lstm_siam
from multi_trainer import Trainer, set_seed
from multi_utils import TemporalProcessor, convert_mctaco_examples_to_features, temprel_set, MATRESProcessor, \
    convert_matres_examples_to_features

from sklearn.model_selection import train_test_split

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


class bigramGetter_fromNN:
    def __init__(self,emb_path,mdl_path,ratio=0.3,layer=1,emb_size=200,splitter=','):
        self.verb_i_map = {}
        f = open(emb_path)
        lines = f.readlines()
        for i,line in enumerate(lines):
            self.verb_i_map[line.split(splitter)[0]] = i
        f.close()
        self.model = VerbNet(len(self.verb_i_map),hidden_ratio=ratio,emb_size=emb_size,num_layers=layer)
        checkpoint = torch.load(mdl_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def eval(self,v1,v2):
        return self.model(torch.from_numpy(np.array([[self.verb_i_map[v1],self.verb_i_map[v2]]])).cuda())

    def getBigramStatsFromTemprel(self,temprel):
        v1,v2='',''
        for i,position in enumerate(temprel.position):
            if position == 'E1':
                v1 = temprel.lemma[i]
            elif position == 'E2':
                v2 = temprel.lemma[i]
                break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.cuda.FloatTensor([0,0]).view(1,-1)
        return torch.cat((self.eval(v1,v2),self.eval(v2,v1)),1).view(1,-1)

    def getBigramStatsFromFeatures(self, cse_position, cse_lemma):
        v1, v2 = '', ''
        for i, position in enumerate(cse_position):
            if position == 'E1':
                v1 = cse_lemma[i]
            elif position == 'E2':
                v2 = cse_lemma[i]
                break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.cuda.FloatTensor([0, 0]).view(1, -1)
        return torch.cat((self.eval(v1, v2), self.eval(v2, v1)), 1).view(1, -1)

    def retrieveEmbeddings(self,temprel):
        v1, v2 = '', ''
        for i, position in enumerate(temprel.position):
            if position == 'E1':
                v1 = temprel.lemma[i]
            elif position == 'E2':
                v2 = temprel.lemma[i]
                break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.zeros_like(self.model.retrieveEmbeddings(torch.from_numpy(np.array([[0,0]])).cuda()).view(1,-1))
        return self.model.retrieveEmbeddings(torch.from_numpy(np.array([[self.verb_i_map[v1],self.verb_i_map[v2]]])).cuda()).view(1,-1)


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

    # hard coded args, fix this later
    # ne_size = 64
    # neg_ratio = 1.0
    train_file = '/home/felix/projects/research/datasets/TBAQ-cleaned/timebank_v4_train.txt'
    dev_file = '/home/felix/projects/research/datasets/TBAQ-cleaned/timebank_v4_dev.txt'

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

    # Prepare MCTACO task
    mctaco_num_labels = 2

    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    matres_train_dataset = None
    mctaco_train_dataset = None
    matres_eval_dataset = None
    mctaco_eval_dataset = None

    # Get MCTACO datasets
    mctaco_processor = TemporalProcessor()
    mctaco_label_list = mctaco_processor.get_labels()

    if training_args.do_train:
        mctaco_train_examples = mctaco_processor.get_train_examples(mctaco_data_args.data_dir)
        mctaco_train_dataset = convert_mctaco_examples_to_features(mctaco_train_examples,
                                                                   mctaco_label_list,
                                                                   mctaco_data_args.max_seq_length,
                                                                   tokenizer)

    if training_args.do_eval or training_args.do_predict:
        mctaco_eval_examples = mctaco_processor.get_dev_examples(mctaco_data_args.data_dir)
        mctaco_eval_dataset = convert_mctaco_examples_to_features(mctaco_eval_examples,
                                                                  mctaco_label_list,
                                                                  mctaco_data_args.max_seq_length,
                                                                  tokenizer)

    # Get MATRES datasets
    matres_processor = MATRESProcessor()
    matres_label_list = matres_processor.get_labels()

    if training_args.do_train:
        matres_train_examples = matres_processor.get_train_examples('')
        matres_train_dataset = convert_matres_examples_to_features(matres_train_examples,
                                                                   matres_label_list,
                                                                   mctaco_data_args.max_seq_length,
                                                                   tokenizer)
    if training_args.do_eval or training_args.do_predict:
        matres_eval_examples = matres_processor.get_dev_examples('')
        matres_eval_dataset = convert_matres_examples_to_features(matres_eval_examples,
                                                                  matres_label_list,
                                                                  mctaco_data_args.max_seq_length,
                                                                  tokenizer)
        # Output Dev gold labels
        eval_labels = []
        for ex in matres_eval_dataset:
            eval_labels.append(ex.label)

        output_eval_label = os.path.join(training_args.output_dir, "test_labels.txt")
        with open(output_eval_label, 'w') as writer:
            for line in eval_labels:
                if line == 0:
                    writer.write("BEFORE\n")
                elif line == 1:
                    writer.write("AFTER\n")
                elif line == 2:
                    writer.write("EQUAL\n")
                elif line == 3:
                    writer.write("VAGUE\n")

    # temprel_trainset = temprel_set("datasets/MATRES/trainset-temprel.xml")
    # temprel_train, temprel_dev = train_test_split(temprel_trainset.temprel_ee, test_size=0.2, random_state=2093)

    # CSE Part
    params = {'embedding_dim':1024,
              'lstm_hidden_dim':64,
              'nn_hidden_dim':64,
              'position_emb_dim':32,
              'bigramStats_dim':2,
              'lemma_emb_dim':200,
              'dropout':False,
              'batch_size':1,
              'granularity':0.2,
              'common_sense_emb_dim':32}

    ratio = 0.3
    emb_size = 200
    layer = 1
    splitter = " "
    print("---------")
    print("ratio=%s,emb_size=%d,layer=%d" % (str(ratio), emb_size, layer))
    emb_path = '../NeuralTemporalRelation-EMNLP19/ser/embeddings_%.1f_%d_%d_timelines.txt' % (ratio, emb_size, layer)
    mdl_path = '../NeuralTemporalRelation-EMNLP19/ser/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)

    bigramGetter = bigramGetter_fromNN(emb_path, mdl_path, ratio, layer, emb_size, splitter=splitter)
    # model = lstm_siam(params, emb_cache, bigramGetter, granularity=0.2,
    #                   common_sense_emb_dim=32, bidirectional=True, lowerCase=False)

    # MCTACO Config, Tokenizer, Model
    mctaco_config = RobertaConfig.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        num_labels=mctaco_num_labels,
        cache_dir=model_args.cache_dir,
        # finetuning_task=mctaco_data_args.task_name,
    )

    multitask_model = RobertaForTemporalMulti.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=mctaco_config,
        cache_dir=model_args.cache_dir,
        params=params,
        bigramGetter=bigramGetter
    )

    # multitask_model.set_batch_size(training_args.per_gpu_train_batch_size)

    def mctaco_compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return glue_compute_metrics("rte", preds, p.label_ids)

    # Initialize our Trainer
    trainer = Trainer(
        model=multitask_model,
        args=training_args,
        timebank_train_dataset=matres_train_dataset,
        mctaco_train_dataset=mctaco_train_dataset,
        eval_dataset=mctaco_eval_dataset, # change to mctaco_eval_dataset / dev_dataset
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
        # preds = trainer.predict(mctaco_eval_dataset)
        # preds = trainer.predict_timebank(dev_dataset)
        preds = trainer.predict(matres_eval_dataset)

        preds_tensor = torch.from_numpy(preds.predictions)
        preds_argmax = torch.argmax(preds_tensor, dim=1)

        # TIMEBANK
        # output_tensor_file = os.path.join(
        #     training_args.output_dir, f"pred_tensor_results_timebank.txt"
        # )
        # with open(output_tensor_file, "w") as tensor_writer:
        #     for val in preds.predictions:
        #         tensor_writer.write(str(val) + "\n")
        # output_pred_file = os.path.join(
        #     training_args.output_dir, f"pred_results_timebank.txt"
        # )
        # with open(output_pred_file, "w") as pred_writer:
        #     for val in preds_argmax:
        #         if val == 0:
        #             pred_writer.write("DATE\n")
        #         elif val == 1:
        #             pred_writer.write("TIME\n")
        #         elif val == 2:
        #             pred_writer.write("DURATION\n")
        #         else:
        #             pred_writer.write("SET\n")

        # MCTACO
        # output_tensor_file = os.path.join(
        #     training_args.output_dir, f"pred_tensor_results_{model_args.model_name_or_path}.txt"
        # )
        # with open(output_tensor_file, "w") as tensor_writer:
        #     for val in preds.predictions:
        #         tensor_writer.write(str(val) + "\n")
        #
        # output_pred_file = os.path.join(
        #     training_args.output_dir, f"pred_results_{model_args.model_name_or_path}.txt"
        # )
        # with open(output_pred_file, "w") as pred_writer:
        #     for val in preds_argmax:
        #         if val == 0:
        #             pred_writer.write("yes\n")
        #         else:
        #             pred_writer.write("no\n")

        # MATRES
        output_tensor_file = os.path.join(
            training_args.output_dir, f"pred_tensor_results_matres_test.txt"
        )
        with open(output_tensor_file, "w") as tensor_writer:
            for val in preds.predictions:
                tensor_writer.write(str(val) + "\n")

        output_pred_file = os.path.join(
            training_args.output_dir, f"pred_results_matres_test.txt"
        )
        with open(output_pred_file, "w") as pred_writer:
            for val in preds_argmax:
                if val == 0:
                    pred_writer.write("BEFORE\n")
                elif val == 1:
                    pred_writer.write("AFTER\n")
                elif val == 2:
                    pred_writer.write("EQUAL\n")
                else:
                    pred_writer.write("VAGUE\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()