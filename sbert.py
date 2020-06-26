"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
Usage:
python training_nli.py
OR
python training_nli.py pretrained_transformer_model_name
"""

import os

from torch import nn
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
# from sentence_transformers.evaluation import LabelAccuracyEvaluator
from CustomLabelAccuracyEvaluator import LabelAccuracyEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import sys

from transformers import DataProcessor


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

    def convert_labels(self, label):
        if label == "yes":
            return 0
        else:
            return 1

    def _create_examples(self, lines, type):
        examples = []
        for (i, line) in enumerate(lines):
            group = line.split("\t")
            guid = "%s-%s" % (type, i)
            text_a = group[0]
            text_b = group[1] + " " + group[2]
            label = self.convert_labels(group[3])
            examples.append(InputExample(guid=guid, texts=[text_a, text_b], label=label))
        return examples


def main():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    # You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

    # Read the dataset
    batch_size = 32
    processor = TemporalProcessor()
    label_list = processor.get_labels()
    train_examples = processor.get_train_examples("/home/felix/projects/research/datasets/MCTACO")
    eval_examples = processor.get_dev_examples("/home/felix/projects/research/datasets/MCTACO")

    # nli_reader = NLIDataReader('../datasets/AllNLI')
    # sts_reader = STSBenchmarkDataReader('../datasets/stsbenchmark')
    train_num_labels = 3
    # model_save_path = 'models/sbert-bert-base-uncased-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = 'models/sbert-roberta-base'

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer("roberta-base")

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read train dataset")
    train_data = SentencesDataset(examples=train_examples, model=model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.SoftmaxLoss(model=model,
                                    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                    num_labels=train_num_labels,
                                    concatenation_sent_rep=True,
                                    concatenation_sent_difference=True,
                                    concatenation_sent_multiplication=False)

    logging.info("Read dev dataset")
    dev_data = SentencesDataset(examples=eval_examples, model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
    evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss)

    # Configure the training
    num_epochs = 3

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path
              )

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    # model = SentenceTransformer(model_save_path)
    # test_data = SentencesDataset(examples=eval_examples, model=model)
    # test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    # evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss)
    #
    # model.evaluate(evaluator)


if __name__ == '__main__':
    main()
