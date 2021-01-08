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
"""PyTorch RoBERTa model. """
import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.configuration_roberta import RobertaConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from transformers.modeling_utils import create_position_ids_from_input_ids

import sympy as sym
import multi_utils_v2

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://cdn.huggingface.co/roberta-base-pytorch_model.bin",
    "roberta-large": "https://cdn.huggingface.co/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://cdn.huggingface.co/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://cdn.huggingface.co/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://cdn.huggingface.co/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://cdn.huggingface.co/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


ROBERTA_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.RobertaTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class RobertaModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """, ROBERTA_START_DOCSTRING)
class RobertaForMaskedLM(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForMaskedLM
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


@add_start_docstrings(
    """RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Roberta Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor`` of shape ``(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForMultipleChoice
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

        """
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Roberta Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForTokenClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForQuestionAnswering(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        # The checkpoint roberta-large is not fine-tuned for question answering. Please see the
        # examples/question-answering/run_squad.py example to see how to fine-tune a model to a question answering task.

        from transformers import RobertaTokenizer, RobertaForQuestionAnswering
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForQuestionAnswering.from_pretrained('roberta-base')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        start_scores, end_scores = model(torch.tensor([input_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


torch.set_default_tensor_type('torch.cuda.FloatTensor')


# class VerbNet(nn.Module):
#     def __init__(self, vocab_size, hidden_ratio=0.5, emb_size=200, num_layers=1):
#         super(VerbNet, self).__init__()
#         self.emb_size = emb_size
#         self.emb_layer = nn.Embedding(vocab_size, self.emb_size)
#         self.fc1 = nn.Linear(self.emb_size*2, int(self.emb_size*2*hidden_ratio))
#         self.num_layers = num_layers
#         if num_layers == 1:
#             self.fc2 = nn.Linear(int(self.emb_size*2*hidden_ratio), 1)
#         else:
#             self.fc2 = nn.Linear(int(self.emb_size*2*hidden_ratio), int(self.emb_size*hidden_ratio))
#             self.fc3 = nn.Linear(int(self.emb_size*hidden_ratio), 1)
#         self.is_training = True
#     def forward(self, x):
#         x_emb = self.emb_layer(x)
#         fullX = torch.cat((x_emb[:,0,:], x_emb[:,1,:]), dim=1)
#         layer1 = F.relu(self.fc1(F.dropout(fullX, p=0.3, training=self.is_training)))
#         if self.num_layers == 1:
#             return torch.sigmoid(self.fc2(layer1))
#         layer2 = F.relu(self.fc2(F.dropout(layer1, p=0.3, training=self.is_training)))
#         layer3 = torch.sigmoid(self.fc3(layer2))
#         return layer3
#     def retrieveEmbeddings(self,x):
#         x_emb = self.emb_layer(x)
#         fullX = torch.cat((x_emb[:, 0, :], x_emb[:, 1, :]), dim=1)
#         layer1 = F.relu(self.fc1(fullX))
#         if self.num_layers == 1:
#             return layer1
#         layer2 = F.relu(self.fc2(layer1))
#         return torch.cat((layer1,layer2),1)
#
#
# class lstm_siam(nn.Module):  # add categorical emb to two places instead of one
#     def __init__(self, params,emb_cache,bigramGetter,granularity=0.05, common_sense_emb_dim=64,bidirectional=False,lowerCase=False):
#         super(lstm_siam, self).__init__()
#         self.params = params
#         self.embedding_dim = params.get('embedding_dim')
#         self.lstm_hidden_dim = params.get('lstm_hidden_dim',64)
#         self.nn_hidden_dim = params.get('nn_hidden_dim',32)
#         self.bigramStats_dim = params.get('bigramStats_dim')
#         self.emb_cache = emb_cache
#         self.bigramGetter = bigramGetter
#         self.output_dim = params.get('output_dim',4)
#         self.batch_size = params.get('batch_size',1)
#         self.granularity = granularity
#         self.common_sense_emb_dim = common_sense_emb_dim
#         self.common_sense_emb = nn.Embedding(int(1.0/self.granularity)*self.bigramStats_dim,self.common_sense_emb_dim)
#         self.bidirectional = bidirectional
#         self.lowerCase = lowerCase
#         if self.bidirectional:
#             self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim // 2,\
#                                 num_layers=1, bidirectional=True)
#         else:
#             self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim,\
#                                 num_layers=1, bidirectional=False)
#         self.h_lstm2h_nn = nn.Linear(2*self.lstm_hidden_dim+self.bigramStats_dim*self.common_sense_emb_dim, self.nn_hidden_dim)
#         self.h_nn2o = nn.Linear(self.nn_hidden_dim+self.bigramStats_dim*self.common_sense_emb_dim, self.output_dim)
#         self.init_hidden()
#     def reset_parameters(self):
#         self.lstm.reset_parameters()
#         self.h_lstm2h_nn.reset_parameters()
#         self.h_nn2o.reset_parameters()
#     def init_hidden(self):
#         if self.bidirectional:
#             self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2),\
#                            torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
#         else:
#             self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),\
#                            torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim))
#
#     def forward(self, temprel):
#         self.init_hidden()
#
#         # common sense embeddings
#         bigramstats = self.bigramGetter.getBigramStatsFromTemprel(temprel)
#         common_sense_emb = self.common_sense_emb(torch.cuda.LongTensor(
#             [min(int(1.0 / self.granularity) - 1, int(bigramstats[0][0] / self.granularity))])).view(1, -1)
#         for i in range(1, self.bigramStats_dim):
#             tmp = self.common_sense_emb(torch.cuda.LongTensor([(i - 1) * int(1.0 / self.granularity) + min(
#                 int(1.0 / self.granularity) - 1, int(bigramstats[0][i] / self.granularity))])).view(1, -1)
#             common_sense_emb = torch.cat((common_sense_emb, tmp), 1)
#
#         if not self.lowerCase:
#             embeds = self.emb_cache.retrieveEmbeddings(tokList=temprel.token).cuda()
#         else:
#             embeds = self.emb_cache.retrieveEmbeddings(tokList=[x.lower() for x in temprel.token]).cuda()
#         embeds = embeds.view(temprel.length,self.batch_size,-1)
#         lstm_out, self.hidden = self.lstm(embeds, self.hidden)
#         lstm_out = lstm_out.view(embeds.size()[0], self.batch_size, self.lstm_hidden_dim)
#         lstm_out = lstm_out[temprel.event_ix][:][:]
#         h_nn = F.relu(self.h_lstm2h_nn(torch.cat((lstm_out.view(1,-1),common_sense_emb),1)))
#         output = self.h_nn2o(torch.cat((h_nn,common_sense_emb),1))
#         return output

def segments_to_index_array_timex(ner_segments):
    per_batch_segments_index = []
    for row in ner_segments:
        per_sentence_segments_index = []
        segments_sequence = []
        for i, val in enumerate(row):
            if val == -95 and (row[i - 1] == -95 or row[i - 1] == -94 or row[i - 1] == -93 or row[i - 1] == -92):
                if segments_sequence:
                    per_sentence_segments_index.append(segments_sequence)
                segments_sequence = []
                segments_sequence.append(i)
            elif val == -95 or val == -94 or val == -93 or val == -92:
                    segments_sequence.append(i)
            else:
                if segments_sequence:
                    per_sentence_segments_index.append(segments_sequence)
                segments_sequence = []
        per_batch_segments_index.append(per_sentence_segments_index)

    return per_batch_segments_index


def segments_to_index_array_event(ner_segments):
    per_batch_segments_index = []
    for row in ner_segments:
        per_sentence_segments_index = []
        segments_sequence = []
        for i, val in enumerate(row):
            if val == -99 and (row[i - 1] == -99 or row[i - 1] == -98 or row[i - 1] == -97 or row[i - 1] == -96):
                if segments_sequence:
                    per_sentence_segments_index.append(segments_sequence)
                segments_sequence = []
                segments_sequence.append(i)
            elif val == -99 or val == -98 or val == -97 or val == -96:
                    segments_sequence.append(i)
            else:
                if segments_sequence:
                    per_sentence_segments_index.append(segments_sequence)
                segments_sequence = []
        per_batch_segments_index.append(per_sentence_segments_index)

    return per_batch_segments_index

class RobertaForTemporalMulti(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, config_2, config_3, config_4, config_5, config_6, config_7, config_8, config_9, config_10, config_11, config_12, config_13, config_14, config_15, config_16, config_17, config_18, config_19, config_20, config_21, config_22, config_23, config_24, config_25, config_26, config_27, config_28, config_29, config_30):
        super().__init__(config, config_2, config_3, config_4, config_5, config_6, config_7, config_8, config_9, config_10, config_11, config_12, config_13, config_14, config_15, config_16, config_17, config_18, config_19, config_20, config_21, config_22, config_23, config_24, config_25, config_26, config_27, config_28, config_29, config_30)
        self.batch_size = 8
        self.num_labels = config.num_labels
        self.num_labels_matres = 4
        self.tb_timex_cls_num_labels = 4
        self.tb_event_cls_num_labels = 7
        self.tb_duration_cls_num_labels = 2
        # self.loss_weight_ratio = {}

        self.roberta = RobertaModel(config)
        self.roberta_2 = RobertaModel(config_2)
        self.roberta_3 = RobertaModel(config_3)
        self.roberta_4 = RobertaModel(config_4)
        self.roberta_5 = RobertaModel(config_5)
        self.roberta_6 = RobertaModel(config_6)
        self.roberta_7 = RobertaModel(config_7)
        self.roberta_8 = RobertaModel(config_8)
        self.roberta_9 = RobertaModel(config_9)
        self.roberta_10 = RobertaModel(config_10)
        self.roberta_11 = RobertaModel(config_11)
        self.roberta_12 = RobertaModel(config_12)
        self.roberta_13 = RobertaModel(config_13)
        self.roberta_14 = RobertaModel(config_14)
        self.roberta_15 = RobertaModel(config_15)
        self.roberta_16 = RobertaModel(config_16)
        self.roberta_17 = RobertaModel(config_17)
        self.roberta_18 = RobertaModel(config_18)
        self.roberta_19 = RobertaModel(config_19)
        self.roberta_20 = RobertaModel(config_20)
        self.roberta_21 = RobertaModel(config_21)
        self.roberta_22 = RobertaModel(config_22)
        self.roberta_23 = RobertaModel(config_23)
        self.roberta_24 = RobertaModel(config_24)
        self.roberta_25 = RobertaModel(config_25)
        self.roberta_26 = RobertaModel(config_26)
        self.roberta_27 = RobertaModel(config_27)
        self.roberta_28 = RobertaModel(config_28)
        self.roberta_29 = RobertaModel(config_29)
        self.roberta_30 = RobertaModel(config_30)

        self.classifier = RobertaClassificationHead(config)

        # self.params = params
        # self.embedding_dim = params.get('embedding_dim', 1024)
        # self.lstm_hidden_dim = params.get('lstm_hidden_dim', 64)
        # self.nn_hidden_dim = params.get('nn_hidden_dim', 64)
        # self.bigramStats_dim = params.get('bigramStats_dim', 2)
        # self.bigramGetter = bigramGetter
        # self.output_dim = params.get('output_dim', 4)
        # self.batch_size = params.get('batch_size', 1)
        # self.granularity = params.get('granularity', 0.2)
        # self.common_sense_emb_dim = params.get('common_sense_emb_dim', 32)
        # self.common_sense_emb = nn.Embedding(int(1.0 / self.granularity) * self.bigramStats_dim, self.common_sense_emb_dim)
        #
        # self.h_lstm2h_nn = nn.Linear(2*self.lstm_hidden_dim+self.bigramStats_dim*self.common_sense_emb_dim, self.nn_hidden_dim)
        # self.h_nn2o = nn.Linear(self.nn_hidden_dim+self.bigramStats_dim*self.common_sense_emb_dim, self.output_dim)
        #
        # self.h_lstm2h_nn_2_cse = nn.Linear(2*1024+self.bigramStats_dim*self.common_sense_emb_dim, 512)
        # self.h_nn2o_2_cse = nn.Linear(512+self.bigramStats_dim*self.common_sense_emb_dim, self.output_dim)

        self.h_lstm2h_nn_2 = nn.Linear(2*1024, 512)
        self.h_nn2o_2 = nn.Linear(512, self.num_labels_matres)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.timebank_timex_classifier = nn.Linear(config.hidden_size, self.tb_timex_cls_num_labels)
        self.timebank_event_classifier = nn.Linear(config.hidden_size, self.tb_event_cls_num_labels)

        self.timebank_duration_classifier = nn.Linear(config.hidden_size, self.tb_duration_cls_num_labels)
        self.init_weights()

    # def reset_parameters(self):
    #     self.h_lstm2h_nn.reset_parameters()
    #     self.h_nn2o.reset_parameters()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def getEventsIdx(self, event_ids):
        E1 = []
        E2 = []
        for i, id in enumerate(event_ids):
            if id == -99:
                E1.append(i)
            elif id == -98:
                E2.append(i)
        return E1, E2

    def getDurationEventId(self, token_index):
        event = []
        for i, id in enumerate(token_index):
            if id == 1:
                event.append(i)
        return event

    # def calculateWeightedLossRatio(self, len_1, len_2, len_3, len_4, len_5):
    #     # task_0_ratio = 0
    #     # task_1_ratio = 0
    #     # task_2_ratio = 0
    #     # task_3_ratio = 0
    #     # task_4_ratio = 0
    #     lengths = [len_1, len_2, len_3, len_4, len_5]
    #     active_lengths = {}
    #     act_lengths = []
    #     for i, length in enumerate(lengths):
    #         if length != 0:
    #             active_lengths[i] = length  # id start at 1
    #             act_lengths.append(length)
    #
    #     n = len(active_lengths)
    #
    #     # fix this to be more efficient
    #     if n == 1:
    #         active_lengths[0] = 1
    #     elif n == 2:
    #         a, b = sym.symbols('a,b')
    #         eq1 = sym.Eq(a + b, 1)
    #         eq2 = sym.Eq(act_lengths[0] * a, act_lengths[1] * b)
    #         result = sym.solve([eq1, eq2], (a, b))
    #         result_list = list(result.values())
    #         for i, key in enumerate(active_lengths):
    #             active_lengths[key] = result_list[i]
    #     elif n == 3:
    #         a, b, c = sym.symbols('a,b,c')
    #         eq1 = sym.Eq(a + b + c, 1)
    #         eq2 = sym.Eq(act_lengths[0] * a, act_lengths[1] * b)
    #         eq3 = sym.Eq(act_lengths[1] * b, act_lengths[2] * c)
    #         result = sym.solve([eq1, eq2, eq3], (a, b, c))
    #         result_list = list(result.values())
    #         for i, key in enumerate(active_lengths):
    #             active_lengths[key] = result_list[i]
    #     elif n == 4:
    #         a, b, c, d = sym.symbols('a,b,c,d')
    #         eq1 = sym.Eq(a + b + c + d, 1)
    #         eq2 = sym.Eq(act_lengths[0] * a, act_lengths[1] * b)
    #         eq3 = sym.Eq(act_lengths[1] * b, act_lengths[2] * c)
    #         eq4 = sym.Eq(act_lengths[2] * c, act_lengths[3] * d)
    #         result = sym.solve([eq1, eq2, eq3, eq4], (a, b, c, d))
    #         result_list = list(result.values())
    #         for i, key in enumerate(active_lengths):
    #             active_lengths[key] = result_list[i]
    #     elif n == 5:
    #         a, b, c, d, e = sym.symbols('a,b,c,d,e')
    #         eq1 = sym.Eq(a + b + c + d + e, 1)
    #         eq2 = sym.Eq(act_lengths[0] * a, act_lengths[1] * b)
    #         eq3 = sym.Eq(act_lengths[1] * b, act_lengths[2] * c)
    #         eq4 = sym.Eq(act_lengths[2] * c, act_lengths[3] * d)
    #         eq5 = sym.Eq(act_lengths[3] * d, act_lengths[4] * e)
    #         result = sym.solve([eq1, eq2, eq3, eq4, eq5], (a, b, c, d, e))
    #         result_list = list(result.values())
    #         for i, key in enumerate(active_lengths):
    #             active_lengths[key] = result_list[i]
    #     else:
    #         active_lengths[0] = 0
    #
    #     # dict = {}
    #     # for i in range(5):
    #     #     for key in active_lengths:
    #     #         if key == i:
    #     #             dict[i] = active_lengths[key]
    #     #         else:
    #     #             dict[i] = 0
    #     # print(dict)
    #
    #     self.loss_weight_ratio = active_lengths

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        event_ids=None,
        # cse_lemma=None,
        # cse_position=None,
        ner_token_ids=None,
        token_index=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_ids=None,
        task_name=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        outputs_2 = self.roberta_2(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        outputs_3 = self.roberta_3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        outputs_4 = self.roberta_4(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        outputs_5 = self.roberta_5(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        outputs_6 = self.roberta_6(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        outputs_7 = self.roberta_7(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        outputs_8 = self.roberta_8(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        outputs_9 = self.roberta_9(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        outputs_10 = self.roberta_10(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # print(labels)
        # print(event_ids)
        # print(task_name)
        if task_name[0] == 1:
            # print("MCTACO")
            sequence_output = outputs[0]
            sequence_output_2 = outputs_2[0]
            # sequence_output_3 = outputs_3[0]
            # sequence_output_4 = outputs_4[0]
            # sequence_output_5 = outputs_5[0]
            # sequence_output_6 = outputs_6[0]
            # sequence_output_7 = outputs_7[0]
            # sequence_output_8 = outputs_8[0]
            # sequence_output_9 = outputs_9[0]
            # sequence_output_10 = outputs_10[0]

            # stacked_sequence_output = torch.mean(torch.stack([sequence_output, sequence_output_2, sequence_output_3, sequence_output_4, sequence_output_5, sequence_output_6, sequence_output_7, sequence_output_8, sequence_output_9, sequence_output_10]), 0)
            # logits = self.classifier(stacked_sequence_output)

            stacked_sequence_output = torch.mean(torch.stack([sequence_output, sequence_output_2]), 0)

            # logits1 = self.classifier(sequence_output)
            # logits2 = self.classifier(sequence_output_2)
            # logits3 = self.classifier(sequence_output_3)
            # logits4 = self.classifier(sequence_output_4)
            # logits5 = self.classifier(sequence_output_5)
            # logits6 = self.classifier(sequence_output_6)
            # logits7 = self.classifier(sequence_output_7)
            # logits8 = self.classifier(sequence_output_8)
            # logits9 = self.classifier(sequence_output_9)
            # logits10 = self.classifier(sequence_output_10)
            #
            # logits = torch.mean(torch.stack(
            #     [logits1,
            #      logits2,
            #      logits3,
            #      logits4,
            #      logits5,
            #      logits6,
            #      logits7,
            #      logits8,
            #      logits9,
            #      logits10]), 0)

            logits = self.classifier(stacked_sequence_output)

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs  # + (self.loss_weight_ratio[0],)
        elif task_name[0] == 2:
            # print("TIMEBANK Time Expression Cls")
            # print(ner_token_ids)
            # print(label_ids)
            segments = segments_to_index_array_timex(ner_token_ids)
            # print(segments)
            sequence_output = outputs[0]
            # print(sequence_output.shape)
            sequence_output = self.dropout(sequence_output)
            # print(sequence_output.shape)

            pooled_seq_output_array = torch.zeros(1, 1024)  # originally (1, 768)
            # pooled_seq_output_array = pooled_seq_output_array.to(device='cuda')
            pooled_seq_output = torch.zeros(1, 1024)  # originally (1, 768)
            # pooled_seq_output = pooled_seq_output.to(device='cuda')

            for row_id in range(self.batch_size):
                if segments[row_id]:
                    for seg in segments[row_id]:
                        pooled_seq_output = torch.zeros(1, 1024)  # originally (1, 768)
                        # pooled_seq_output = pooled_seq_output.to(device='cuda')
                        for idx in seg:
                            pooled_seq_output += sequence_output[row_id][idx]
                        pooled_seq_output = pooled_seq_output / len(seg)
                        pooled_seq_output_array = torch.cat([pooled_seq_output_array, pooled_seq_output], dim=0)
            # print(pooled_seq_output_array.shape)
            pooled_seq_output_array = pooled_seq_output_array[1:].unsqueeze(dim=0)
            # print(pooled_seq_output_array.shape)
            logits = self.timebank_timex_classifier(pooled_seq_output_array)
            # print(logits)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here\
            # print(label_ids)
            # print(labels)
            # print(len(label_ids))
            # print(len(labels))
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                active_logits = logits.view(-1, self.tb_timex_cls_num_labels)
                # print(active_logits)
                active_labels = []
                for lab in labels.view(-1):
                    if lab != -100:
                        active_labels.append(lab.item())
                # print(active_labels)
                active_labels = torch.tensor(active_labels).type_as(labels)
                # print(active_labels)
                # active_labels = active_labels.to(device='cuda')
                loss = loss_fct(active_logits, active_labels)
                outputs = (loss,) + outputs  # + (self.loss_weight_ratio[1],)
        elif task_name[0] == 3:
            # print("TIMEBANK Event Cls")
            segments = segments_to_index_array_event(ner_token_ids)
            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)

            pooled_seq_output_array = torch.zeros(1, 1024)  # originally (1, 768)
            # pooled_seq_output_array = pooled_seq_output_array.to(device='cuda')
            pooled_seq_output = torch.zeros(1, 1024)  # originally (1, 768)
            # pooled_seq_output = pooled_seq_output.to(device='cuda')

            for row_id in range(self.batch_size):
                if segments[row_id]:
                    for seg in segments[row_id]:
                        pooled_seq_output = torch.zeros(1, 1024)  # originally (1, 768)
                        # pooled_seq_output = pooled_seq_output.to(device='cuda')
                        for idx in seg:
                            pooled_seq_output += sequence_output[row_id][idx]
                        pooled_seq_output = pooled_seq_output / len(seg)
                        pooled_seq_output_array = torch.cat([pooled_seq_output_array, pooled_seq_output], dim=0)

            pooled_seq_output_array = pooled_seq_output_array[1:].unsqueeze(dim=0)
            logits = self.timebank_event_classifier(pooled_seq_output_array)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                active_logits = logits.view(-1, self.tb_event_cls_num_labels)
                active_labels = []
                for lab in labels.view(-1):
                    if lab != -100:
                        active_labels.append(lab.item())
                active_labels = torch.tensor(active_labels).type_as(labels)
                # active_labels = active_labels.to(device='cuda')
                loss = loss_fct(active_logits, active_labels)
                outputs = (loss,) + outputs  # + (self.loss_weight_ratio[2],)
        elif task_name[0] == 4:
            # print("MATRES")
            sequence_output = outputs[0]
            # print(len(sequence_output))
            # print(len(sequence_output[0]))
            # print(len(sequence_output[0][0]))
            # print(len(outputs[1]))
            # print(len(outputs[1][0]))

            e1_hidden_mean_batch = []
            e2_hidden_mean_batch = []
            for i in range(len(sequence_output)):
                e1,e2 = self.getEventsIdx(event_ids[i])
                list_e1 = []
                list_e2 = []
                for j in e1:
                    list_e1.append(sequence_output[i][j])
                for k in e2:
                    list_e2.append(sequence_output[i][k])

                e1_hidden_mean = torch.mean(torch.stack(list_e1), 0)
                e2_hidden_mean = torch.mean(torch.stack(list_e2), 0)

                e1_hidden_mean_batch.append(e1_hidden_mean)
                e2_hidden_mean_batch.append(e2_hidden_mean)

            e1_hidden_mean_batch_stk = torch.stack(e1_hidden_mean_batch)
            e2_hidden_mean_batch_stk = torch.stack(e2_hidden_mean_batch)

            events_concatenated = torch.cat((e1_hidden_mean_batch_stk, e2_hidden_mean_batch_stk), 1)

            # # common sense embeddings
            # bigramstats = self.bigramGetter.getBigramStatsFromFeatures(cse_position, cse_lemma)
            # common_sense_emb = self.common_sense_emb(torch.cuda.LongTensor(
            #     [min(int(1.0 / self.granularity) - 1, int(bigramstats[0][0] / self.granularity))])).view(1, -1)
            # for i in range(1, self.bigramStats_dim):
            #     tmp = self.common_sense_emb(torch.cuda.LongTensor([(i - 1) * int(1.0 / self.granularity) + min(
            #         int(1.0 / self.granularity) - 1, int(bigramstats[0][i] / self.granularity))])).view(1, -1)
            #     common_sense_emb = torch.cat((common_sense_emb, tmp), 1)
            #
            # h_nn = F.relu(self.h_lstm2h_nn_2_cse(torch.cat((events_concatenated, common_sense_emb), 1)))
            # output_logits = self.h_nn2o_2_cse(torch.cat((h_nn, common_sense_emb), 1))

            h_nn = F.relu(self.h_lstm2h_nn_2(events_concatenated))
            output_logits = self.h_nn2o_2(h_nn)

            outputs = (output_logits,) + outputs[2:]
            if labels is not None:
                if self.num_labels_matres == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(output_logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(output_logits.view(-1, self.num_labels_matres), labels.view(-1))
                outputs = (loss,) + outputs  # + (self.loss_weight_ratio[3],)
        else:
            # print("TB Duration")
            sequence_output = outputs[0]

            event_hidden_mean_batch = []
            for i in range(len(sequence_output)):
                event = self.getDurationEventId(token_index[i])
                list_event = []
                for j in event:
                    list_event.append(sequence_output[i][j])

                event_hidden_mean = torch.mean(torch.stack(list_event), 0)

                event_hidden_mean_batch.append(event_hidden_mean)

            event_hidden_mean_batch_stk = torch.stack(event_hidden_mean_batch)

            pooled_output = self.dropout(event_hidden_mean_batch_stk)
            logits = self.timebank_duration_classifier(pooled_output)

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                if self.tb_duration_cls_num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.tb_duration_cls_num_labels), labels.view(-1))
                outputs = (loss,) + outputs  # + (self.loss_weight_ratio[4],)

        return outputs  # (loss), logits, (hidden_states), (attentions)
