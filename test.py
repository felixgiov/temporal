# from transformers import BertTokenizer
# from timebank_modeling_bert import BertForTokenEventClassification, BertConfig
# import torch
#
# config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForTokenEventClassification.from_pretrained('bert-base-uncased', config=config)
#
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# labels = torch.tensor([-100, 0, 0, 0, 1, 0, 2, -100]).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids, labels=labels)
#
# loss, scores = outputs[:2]
#
# print(tokenizer.tokenize("Hello, my dog is cute"))
# print(tokenizer.encode("Hello, my dog is cute", add_special_tokens=False))
# print(input_ids)
# print(torch.tensor([1] * input_ids.size(1)))
# print(labels)
# print(loss)
# print(scores)

# torch.Size([1, 8, 768])                                                           sequence output
# tensor([[[-0.1144,  0.1937,  0.1250,  ..., -0.3827,  0.2107,  0.5407],
#          [ 0.5308,  0.3207,  0.3665,  ..., -0.0036,  0.7579,  0.0388],
#          [-0.4877,  0.8849,  0.4256,  ..., -0.6976,  0.4458,  0.1231],
#          ...,
#          [-0.7003, -0.1815,  0.3297,  ..., -0.4838,  0.0680,  0.8901],
#          [-1.0355, -0.2567, -0.0317,  ...,  0.3197,  0.3999,  0.1795],
#          [ 0.6080,  0.2610, -0.3131,  ...,  0.0311, -0.6283, -0.1994]]],
#        grad_fn=<NativeLayerNormBackward>)
# torch.Size([1, 8, 768])                                                          sequence output after dropout
# tensor([[[-0.1144,  0.1937,  0.1250,  ..., -0.3827,  0.2107,  0.5407],
#          [ 0.5308,  0.3207,  0.3665,  ..., -0.0036,  0.7579,  0.0388],
#          [-0.4877,  0.8849,  0.4256,  ..., -0.6976,  0.4458,  0.1231],
#          ...,
#          [-0.7003, -0.1815,  0.3297,  ..., -0.4838,  0.0680,  0.8901],
#          [-1.0355, -0.2567, -0.0317,  ...,  0.3197,  0.3999,  0.1795],
#          [ 0.6080,  0.2610, -0.3131,  ...,  0.0311, -0.6283, -0.1994]]],
#        grad_fn=<NativeLayerNormBackward>)
# torch.Size([1, 8, 3])                                                            logits
# tensor([[[ 0.9780, -0.1662, -0.1183],
#          [ 0.2666,  0.3688,  0.0591],
#          [ 0.1719, -0.0542, -0.0373],
#          [ 0.2429,  0.0533,  0.4143],
#          [ 0.0219, -0.0160,  0.1718],
#          [-0.6171,  0.1116,  0.4703],
#          [-0.0149,  0.4675,  0.7406],
#          [-0.0251, -0.2628,  0.5650]]], grad_fn=<AddBackward0>)
# ['hello', ',', 'my', 'dog', 'is', 'cute']
# [7592, 1010, 2026, 3899, 2003, 10140]
# tensor([[  101,  7592,  1010,  2026,  3899,  2003, 10140,   102]])
# tensor([1, 1, 1, 1, 1, 1, 1, 1])
# tensor([[-100,    0,    0,    0,    1,    0,    2, -100]])
# tensor(1.1519, grad_fn=<NllLossBackward>)
# tensor([[[ 0.9780, -0.1662, -0.1183],
#          [ 0.2666,  0.3688,  0.0591],
#          [ 0.1719, -0.0542, -0.0373],
#          [ 0.2429,  0.0533,  0.4143],
#          [ 0.0219, -0.0160,  0.1718],
#          [-0.6171,  0.1116,  0.4703],
#          [-0.0149,  0.4675,  0.7406],
#          [-0.0251, -0.2628,  0.5650]]], grad_fn=<AddBackward0>)


# samples = torch.cat((-torch.ones(5), torch.ones(5)))
# samples2 = torch.cat((torch.ones(50) * 5, torch.ones(5) * -5))
#
# print(samples)
# print(samples2)

# import logging
# logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
# logging.info('Finished tokenizing sentences!')

# count = 0
# just_dash_count = 0
# with open('/home/felix/projects/research/datasets/TBAQ-cleaned/timebank_v2.txt', 'r') as reader:
#     for i, line in enumerate(reader):
#         if not line.startswith("#"):
#             items = line.split("\t")
#             if "-" in items[1]:
#                 print(str(i)+" "+line)
#                 count += 1
#                 if items[1] == "-":
#                     just_dash_count += 1
# print(just_dash_count)
# print(count)

# from nltk.tokenize.stanford import StanfordTokenizer
# jar = '/home/felix/tools/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar'
# tk = StanfordTokenizer(jar)
# tokenized = tk.tokenize("In Washington, the US State Department issued a statement regretting \"the untimely death\" of the rapier-tongued Scottish barrister and parliamentarian.")
# print(tokenized)

# if "BEFORE" in ['BEFORE', 'AFTER', 'INCLUDES', 'IS_INCLUDED', 'DURING',
#                    'SIMULTANEOUS', 'IAFTER', 'IBEFORE', 'IDENTITY', 'BEGINS',
#                    'ENDS', 'BEGUN_BY', 'ENDED_BY', 'DURING_INV']:
#     print("YAS")
# else:
#     print("NOOO")

"""multi_run.py"""
# Timebank Relation parts
# from collections import defaultdict
#
# rel_count = defaultdict(lambda: 0)
#
# _, train_toks, train_labs, _, train_rels, bio2ix, ne2ix, _, rel2ix = multi_utils_v2.extract_rel_data_from_mh_conll_v2(train_file, neg_ratio)
# print(bio2ix)
# print(ne2ix)
# print(rel2ix)
# print('max sent len:', multi_utils_v2.max_sents_len(train_toks, tokenizer))
# print(min([len(sent_rels) for sent_rels in train_rels]), max([len(sent_rels) for sent_rels in train_rels]))
# print()
#
# _, dev_toks, dev_labs, _, dev_rels, _, _, _, _ = multi_utils_v2.extract_rel_data_from_mh_conll_v2(dev_file, 1.0)
# print('max sent len:', multi_utils_v2.max_sents_len(dev_toks, tokenizer))
# print(min([len(sent_rels) for sent_rels in dev_rels]), max([len(sent_rels) for sent_rels in dev_rels]))
# print()
#
#
# for sent_rels in train_rels:
#     for rel in sent_rels:
#         rel_count[rel[-1]] += 1
#
# for sent_rels in dev_rels:
#     for rel in sent_rels:
#         rel_count[rel[-1]] += 1
#
# print(rel_count)
#
# max_len = max(
#     multi_utils_v2.max_sents_len(train_toks, tokenizer),
#     multi_utils_v2.max_sents_len(dev_toks, tokenizer),
# )
#
# train_dataset = multi_utils_v2.convert_rels_to_tensors(train_toks, train_labs, train_rels, tokenizer, bio2ix, ne2ix, rel2ix,
#                                               max_len)
# dev_dataset = multi_utils_v2.convert_rels_to_tensors(dev_toks, dev_labs, dev_rels, tokenizer, bio2ix, ne2ix, rel2ix, max_len)
#
# # train_sampler = RandomSampler(train_dataset)
# # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
# # dev_sampler = SequentialSampler(dev_dataset)
# # dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.batch_size)

# #TimeBank Joint
# from collections import defaultdict
#
# cls_tok = '<s>' if isinstance(tokenizer, RobertaTokenizer) else '[CLS]'
# sep_tok = '</s>' if isinstance(tokenizer, RobertaTokenizer) else '[SEP]'
# pad_tok = '<pad>' if isinstance(tokenizer, RobertaTokenizer) else '[PAD]'
# unk_tok = '<unk>' if isinstance(tokenizer, RobertaTokenizer) else '[UNK]'
#
# print(f"Specal Tokens: {cls_tok}, {sep_tok}, {pad_tok}, {unk_tok}")
#
# train_comments, train_toks, train_ners, train_mods, train_rels, bio2ix, ne2ix, mod2ix, rel2ix = multi_utils_v2.extract_rel_data_from_mh_conll_v2(
#     train_file,
#     down_neg=0.0
# )
#
# print(bio2ix)
# print(ne2ix)
# print(rel2ix)
# print(mod2ix)
# print()
#
# print('max sent len:', multi_utils_v2.max_sents_len(train_toks, tokenizer))
# print(min([len(sent_rels) for sent_rels in train_rels]), max([len(sent_rels) for sent_rels in train_rels]))
# print()
#
# dev_comments, dev_toks, dev_ners, dev_mods, dev_rels, _, _, _, _ = multi_utils_v2.extract_rel_data_from_mh_conll_v2(
#     args.dev_file, down_neg=0.0)
# print('max sent len:', multi_utils_v2.max_sents_len(dev_toks, tokenizer))
# print(min([len(sent_rels) for sent_rels in dev_rels]), max([len(sent_rels) for sent_rels in dev_rels]))
# print()
#
# rel_count = defaultdict(lambda: 0)
#
# for sent_rels in train_rels:
#     for rel in sent_rels:
#         rel_count[rel[-1]] += 1
#
# for sent_rels in dev_rels:
#     for rel in sent_rels:
#         rel_count[rel[-1]] += 1
#
# print(rel_count)
#
# example_id = 15
# print('Random example: id %i, len: %i' % (example_id, len(train_toks[example_id])))
# for tok_id in range(len(train_toks[example_id])):
#     print("%i\t%10s\t%s" % (tok_id, train_toks[example_id][tok_id], train_ners[example_id][tok_id]))
# print(train_rels[example_id])
# print()
#
# max_len = max(
#     multi_utils_v2.max_sents_len(train_toks, tokenizer),
#     multi_utils_v2.max_sents_len(dev_toks, tokenizer),
# )
# cls_max_len = max_len + 2
#
# train_dataset, train_tok, train_ner, train_mod, train_rel, train_spo = multi_utils_v2.convert_rels_to_mhs_v3(
#     train_toks, train_ners, train_mods, train_rels,
#     tokenizer, bio2ix, mod2ix, rel2ix, max_len,
#     cls_tok=cls_tok, sep_tok=sep_tok, pad_tok=pad_tok, deunk=False, verbose=0)
#
# dev_dataset, dev_tok, dev_ner, dev_mod, dev_rel, dev_spo = multi_utils_v2.convert_rels_to_mhs_v3(
#     dev_toks, dev_ners, dev_mods, dev_rels,
#     tokenizer, bio2ix, mod2ix, rel2ix, max_len,
#     cls_tok=cls_tok, sep_tok=sep_tok, pad_tok=pad_tok, deunk=False, verbose=0)
#
# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
#
# num_epoch_steps = len(train_dataloader)
# num_training_steps = args.num_epoch * num_epoch_steps
# save_step_interval = math.ceil(num_epoch_steps / args.save_step_portion)
#
# model = JointNerModReExtractor(
#     bert_url=args.pretrained_model,
#     ner_emb_size=bio_emb_size, ner_vocab=bio2ix,
#     mod_emb_size=mod_emb_size, mod_vocab=mod2ix,
#     rel_emb_size=rel_emb_size, rel_vocab=rel2ix,
#     device=args.device
# )

"""multi_modelling_roberta.py"""
# class RobertaRel(BertPreTrainedModel):
#
#     def __init__(self, config, ne_size, num_ne, num_rel):
#         super(RobertaRel, self).__init__(config)
#         self.num_rel = num_rel
#         self.num_ne = num_ne
#         self.ne_size = ne_size
#         self.roberta = RobertaModel(config)
#         if ne_size:
#             self.ne_embed = nn.Embedding(num_ne, ne_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.head_mat = nn.Linear(config.hidden_size + ne_size,
#                                   config.hidden_size + ne_size, bias=False)
#         self.tail_mat = nn.Linear(config.hidden_size + ne_size,
#                                   config.hidden_size + ne_size, bias=False)
#         self.h2o = nn.Linear(2 * config.hidden_size + 2 * ne_size, num_rel)
#         self.init_weights()
#
#     def forward(self, tok_ix, attn_mask, tail_mask, tail_labs, head_mask, head_labs, rel_labs=None):
#         # import pdb; pdb.set_trace()
#         encoder_out = self.roberta(tok_ix, attention_mask=attn_mask)[0]
#         tail_rep = torch.bmm(tail_mask.unsqueeze(1).float(), encoder_out).squeeze(1)
#         head_rep = torch.bmm(head_mask.unsqueeze(1).float(), encoder_out).squeeze(1)
#         if self.ne_size:
#             tail_ne = self.ne_embed(tail_labs)
#             head_ne = self.ne_embed(head_labs)
#             tail_rep = torch.cat((tail_rep, tail_ne), dim=-1)
#             head_rep = torch.cat((head_rep, head_ne), dim=-1)
#
#         concat_out = self.dropout(F.relu(torch.cat((self.tail_mat(tail_rep), self.head_mat(head_rep)), dim=-1)))
#         logits = self.h2o(concat_out)
#         outputs = (logits, )
#
#         if rel_labs is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_rel), rel_labs.view(-1))
#             outputs = (loss, ) + outputs
#
#         return outputs  # (loss), logits, (hidden_states), (attentions)
#
#
# class RobertaForTemporalMultitask(BertPreTrainedModel):
#     config_class = RobertaConfig
#     pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
#     base_model_prefix = "roberta"
#
#     def __init__(self, config, ne_size, num_ne, num_rel):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#
#         self.roberta = RobertaModel(config)
#         self.classifier = RobertaClassificationHead(config)
#
#         self.num_rel = num_rel
#         self.num_ne = num_ne
#         self.ne_size = ne_size
#
#         if ne_size:
#             self.ne_embed = nn.Embedding(num_ne, ne_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.head_mat = nn.Linear(config.hidden_size + ne_size,
#                                   config.hidden_size + ne_size, bias=False)
#         self.tail_mat = nn.Linear(config.hidden_size + ne_size,
#                                   config.hidden_size + ne_size, bias=False)
#         self.h2o = nn.Linear(2 * config.hidden_size + 2 * ne_size, num_rel)
#         self.init_weights()
#
#     @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         tok_ix, attn_mask, tail_mask, tail_labs, head_mask, head_labs,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         # head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         rel_labs=None,
#         task_name=None
#     ):
#         if task_name == 0:
#             outputs = self.roberta(
#                 input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 position_ids=position_ids,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#             )
#             sequence_output = outputs[0]
#             logits = self.classifier(sequence_output)
#
#             outputs = (logits,) + outputs[2:]
#             if labels is not None:
#                 if self.num_labels == 1:
#                     #  We are doing regression
#                     loss_fct = MSELoss()
#                     loss = loss_fct(logits.view(-1), labels.view(-1))
#                 else:
#                     loss_fct = CrossEntropyLoss()
#                     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#                 outputs = (loss,) + outputs
#
#             return outputs  # (loss), logits, (hidden_states), (attentions)
#
#         else:
#             # import pdb; pdb.set_trace()
#             encoder_out = self.roberta(tok_ix, attention_mask=attn_mask)[0]
#             tail_rep = torch.bmm(tail_mask.unsqueeze(1).float(), encoder_out).squeeze(1)
#             head_rep = torch.bmm(head_mask.unsqueeze(1).float(), encoder_out).squeeze(1)
#             if self.ne_size:
#                 tail_ne = self.ne_embed(tail_labs)
#                 head_ne = self.ne_embed(head_labs)
#                 tail_rep = torch.cat((tail_rep, tail_ne), dim=-1)
#                 head_rep = torch.cat((head_rep, head_ne), dim=-1)
#
#             concat_out = self.dropout(F.relu(torch.cat((self.tail_mat(tail_rep), self.head_mat(head_rep)), dim=-1)))
#             logits = self.h2o(concat_out)
#             outputs = (logits,)
#
#             if rel_labs is not None:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_rel), rel_labs.view(-1))
#                 outputs = (loss,) + outputs
#
#             return outputs  # (loss), logits, (hidden_states), (attentions)
#
#
# class JointNerModReExtractor(nn.Module):
#     def __init__(self, bert_url,
#                  ner_emb_size, ner_vocab,
#                  mod_emb_size, mod_vocab,
#                  rel_emb_size, rel_vocab,
#                  hidden_size=768, device=None):
#         super(JointNerModReExtractor, self).__init__()
#
#         self.ner_vocab = ner_vocab
#         self.mod_vocab = mod_vocab
#         self.rel_vocab = rel_vocab
#
#         self.device = device
#
#         self.ner_emb = nn.Embedding(num_embeddings=len(ner_vocab), embedding_dim=ner_emb_size)
#         self.mod_emb = nn.Embedding(num_embeddings=len(mod_vocab), embedding_dim=mod_emb_size)
#         self.rel_emb = nn.Embedding(num_embeddings=len(rel_vocab), embedding_dim=rel_emb_size)
#
#         self.encoder = BertModel.from_pretrained(bert_url, output_hidden_states=True)
#
#         self.activation = nn.Tanh()
#
#         self.crf_tagger = CRF(len(ner_vocab), batch_first=True)
#
#         self.crf_emission = nn.Linear(hidden_size, len(ner_vocab))
#
#         self.mod_h2o = nn.Linear(hidden_size + ner_emb_size, len(mod_vocab))
#         self.mod_loss_func = nn.CrossEntropyLoss(reduction='none')
#
#         # self.mhs_u = nn.Linear(hidden_size + ner_emb_size + mod_emb_size,
#         #                        rel_emb_size, bias=False)
#         # self.mhs_v = nn.Linear(hidden_size + ner_emb_size + mod_emb_size,
#         #                        rel_emb_size, bias=False)
#
#         self.sel_u_mat = nn.Parameter(torch.Tensor(rel_emb_size, hidden_size + ner_emb_size + mod_emb_size))
#         nn.init.kaiming_uniform_(self.sel_u_mat, a=math.sqrt(5))
#
#         self.sel_v_mat = nn.Parameter(torch.Tensor(rel_emb_size, hidden_size + ner_emb_size + mod_emb_size))
#         nn.init.kaiming_uniform_(self.sel_v_mat, a=math.sqrt(5))
#
#         self.drop_uv = nn.Dropout(p=0.1)
#         self.rel_h2o = nn.Linear(rel_emb_size, len(rel_vocab), bias=False)
#
#         self.id2ner = {v: k for k, v in self.ner_vocab.items()}
#         self.id2mod = {v: k for k, v in self.mod_vocab.items()}
#         self.id2rel = {v: k for k, v in self.rel_vocab.items()}
#
#     def forward(self, tokens, mask, ner_gold=None, mod_gold=None, rel_gold=None, reduction='token_mean'):
#
#         # output tuple
#         loss_outputs = ()
#         pred_outputs = ()
#
#         batch_size, seq_len = tokens.shape
#         _, _, all_hiddens = self.encoder(tokens, attention_mask=mask)  # last hidden of BERT
#         low_o = all_hiddens[6]
#         high_o = all_hiddens[12]
#
#         ner_logits = self.crf_emission(low_o)
#
#         # ner section
#         if all(gold is not None for gold in [ner_gold, mod_gold, rel_gold]):
#             crf_loss = -self.crf_tagger(ner_logits, ner_gold,
#                                         mask=mask,
#                                         reduction=reduction)
#             loss_outputs += (crf_loss,)
#         else:
#             decoded_ner_ix = self.crf_tagger.decode(emissions=ner_logits, mask=mask)
#             decoded_ner_tags = [list(map(lambda x: self.id2ner[x], tags)) for tags in decoded_ner_ix]
#             pred_outputs += (decoded_ner_tags,)
#             temp_tag = copy.deepcopy(decoded_ner_ix)
#             for line in temp_tag:
#                 line.extend([self.ner_vocab['O']] * (seq_len - len(line)))
#             ner_gold = torch.tensor(temp_tag).to(self.device)
#
#         ner_out = self.ner_emb(ner_gold)
#         o = torch.cat((low_o, ner_out), dim=2)
#
#         # mod section
#         mod_logits = self.mod_h2o(o)
#         if all(gold is not None for gold in [ner_gold, mod_gold, rel_gold]):
#             mod_loss = self.mod_loss_func(mod_logits.view(-1, len(self.mod_vocab)), mod_gold.view(-1))
#             mod_loss = mod_loss.masked_select(mask.view(-1)).sum()/mask.sum()
#             loss_outputs += (mod_loss,)
#         else:
#             pred_mod = mod_logits.argmax(-1)
#             decoded_mod = multi_utils_v2.decode_tensor_prediction(pred_mod, mask)
#             pred_outputs += ([list(map(lambda x: self.id2mod[x], mod)) for mod in decoded_mod],)
#             mod_gold = pred_mod
#
#         mod_out = self.mod_emb(mod_gold)
#         o = torch.cat((high_o, ner_out, mod_out), dim=-1)
#
#         # forward multi head selection
#         # u = self.mhs_u(o).unsqueeze(1).expand(batch_size, seq_len, seq_len, -1)
#         # v = self.mhs_v(o).unsqueeze(2).expand(batch_size, seq_len, seq_len, -1)
#         # uv = self.activation(u + v)
#         # uv = self.activation(torch.cat((u, v, (u - v).abs()), dim=-1))
#         # # correct one
#
#         '''Multi-head Selection'''
#         # word representations: [b, l, r_s]
#         # broadcast sum: [b, l, 1, h] + [b, 1, l, h] = [b, l, l, h]
#         u = o.matmul(self.sel_u_mat.t())  # [b, l, h_s] -> [b, l, r_s]
#         v = o.matmul(self.sel_v_mat.t())  # [b, l, h_s] -> [b, l, r_s]
#         uv = self.activation(u.unsqueeze(2) + v.unsqueeze(1))
#         uv = self.drop_uv(uv)
#         # rel_logits = torch.einsum('bijh,rh->birj', [uv, self.relation_emb.weight])
#         rel_logits = self.rel_h2o(uv).transpose(2, 3)
#
#         if all(gold is not None for gold in [ner_gold, mod_gold, rel_gold]):
#             rel_loss = self.masked_BCEloss(
#                 rel_logits,
#                 rel_gold,
#                 mask,
#                 reduction
#             )
#             loss_outputs += (rel_loss,)
#         else:
#             rel_ix_triplets = self.inference(mask, decoded_ner_tags, rel_logits, self.id2rel)
#             pred_outputs += (rel_ix_triplets,)
#
#         return loss_outputs + pred_outputs
#
#     @staticmethod
#     def description(epoch, epoch_num, output):
#         return f"L: {output['loss'].item():.6f}, L_ner: {output['crf_loss'].item():.6f}, " \
#                f"L_mod: {output['mod_loss'].item():.6f}, L_rel: {output['selection_loss'].item():.6f}, " \
#                f"epoch: {epoch}/{epoch_num}:"
#
#     @staticmethod
#     def masked_BCEloss(selection_logits, selection_gold, mask, reduction):
#         _, _, rel_size, _ = selection_logits.shape
#         # batch x seq x rel x seq
#         selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, rel_size, -1)
#         selection_loss = F.binary_cross_entropy_with_logits(selection_logits, selection_gold, reduction='none')
#         selection_loss = selection_loss.masked_select(selection_mask).sum()
#         if reduction in ['token_mean']:
#             selection_loss /= mask.sum()
#         return selection_loss
#
#     @staticmethod
#     def selection_decode(ner_tags, selection_tags, id2rel):
#
#         def find_entity(pos, s_ner_tags):
#             entity = []
#
#             if s_ner_tags[pos][0] in ['B', 'O']:
#                 entity.append(pos)
#             else:
#                 temp_entity = []
#                 while s_ner_tags[pos][0] == 'I':
#                     temp_entity.append(pos)
#                     pos -= 1
#                     if pos < 0:
#                         break
#                     if s_ner_tags[pos][0] == 'B':
#                         temp_entity.append(pos)
#                         break
#                 entity = list(reversed(temp_entity))
#             return entity
#
#         batch_num = len(ner_tags)
#         rel_ix_result = [[] for _ in range(batch_num)]
#         idx = torch.nonzero(selection_tags.cpu())
#
#         for i in range(idx.size(0)):
#             b, s, p, o = idx[i].tolist()
#
#             predicate = id2rel[p]
#             if predicate == 'N':
#                 continue
#             tags = ner_tags[b]
#             object_ix = find_entity(o, tags)
#             subject_ix = find_entity(s, tags)
#             assert object_ix != [] and subject_ix != []
#
#             rel_ix_triplet = {
#                 'subject': subject_ix,
#                 'predicate': predicate,
#                 'object': object_ix
#             }
#             rel_ix_result[b].append(rel_ix_triplet)
#         return rel_ix_result
#
#     @staticmethod
#     def inference(mask, decoded_tag, selection_logits, id2rel):
#         # mask: B x L x R x L
#         _, _, rel_size, _ = selection_logits.shape
#
#         selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, rel_size, -1)
#         selection_tags = (torch.sigmoid(selection_logits) * selection_mask.float()) > 0.5
#         selection_triplets = JointNerModReExtractor.selection_decode(decoded_tag, selection_tags, id2rel)
#         return selection_triplets