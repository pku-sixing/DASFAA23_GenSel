import argparse
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config

from FineUniGPT import FineUniGPT
from GPTModels import GPT2HybridLMHeadModel
from train_FineUniGPT import load_dataset
from utils import top_k_top_p_filtering, set_logger
from vocab.special_vocab import get_tokenizer, type2id, source2id, segment2id, csk_relation_set


def collate_fn(batch):
    # input_ids, context_length, full_length
    raw_input_ids = [x[0] for x in batch]
    context_lengths = [x[1] for x in batch]
    segment_ids = [x[3] for x in batch]
    source_ids = [x[4] for x in batch]
    source_indices = [x[5] for x in batch]
    type_ids = [x[6] for x in batch]
    fw_pos = [x[7] for x in batch]
    bw_pos = [x[8] for x in batch]
    word_align = [x[9] for x in batch]
    word_index_fw = [x[10] for x in batch]
    word_index_bw = [x[11] for x in batch]

    input_ids = rnn_utils.pad_sequence(raw_input_ids, batch_first=True, padding_value=5)
    labels = rnn_utils.pad_sequence(raw_input_ids, batch_first=True, padding_value=-100)
    segment_ids = rnn_utils.pad_sequence(segment_ids, batch_first=True, padding_value=0)
    source_ids = rnn_utils.pad_sequence(source_ids, batch_first=True, padding_value=0)
    source_indices = rnn_utils.pad_sequence(source_indices, batch_first=True, padding_value=0)
    fw_pos = rnn_utils.pad_sequence(fw_pos, batch_first=True, padding_value=0)
    bw_pos = rnn_utils.pad_sequence(bw_pos, batch_first=True, padding_value=0)
    type_ids = rnn_utils.pad_sequence(type_ids, batch_first=True, padding_value=0)
    word_align = rnn_utils.pad_sequence(word_align, batch_first=True, padding_value=0)
    word_index_fw = rnn_utils.pad_sequence(word_index_fw, batch_first=True, padding_value=0)
    word_index_bw = rnn_utils.pad_sequence(word_index_bw, batch_first=True, padding_value=0)
    context_lengths = torch.stack(context_lengths)
    return input_ids, labels, segment_ids, source_ids, source_indices, type_ids, fw_pos, bw_pos, \
           word_align, word_index_fw, word_index_bw, context_lengths


def generate_next_token(args, model, instance, unk_id, eos_id, query_length, repetition_penalty,
                        ending_penalty, target_length, num_samples, beam_mode=False):
    """
    对于给定的上文，生成下一个单词
    """
    outputs = model.forward(instance['input_ids'],
                            segments=instance['segments'],
                            sources=instance['sources'],
                            source_index=instance['source_index'],
                            token_types=instance['token_types'],
                            pos_fw=instance['pos_fw'],
                            pos_bw=instance['pos_bw'],
                            word_aligns=instance['word_aligns'],
                            word_aligns_fw=instance['word_aligns_fw'],
                            word_aligns_bw=instance['word_aligns_bw'],
                            attention_mask=None,
                            labels=None,
                            past_key_values=instance.get('past_key_values', None),
                            )

    logits = outputs.logits
    past_key_values = outputs.past_key_values
    # next_token_logits表示最后一个token的hidden_state对应的prediction_scores,也就是模型要预测的下一个token的概率
    next_token_logits = logits[0, -1, :]
    raw_probs = F.softmax(next_token_logits, dim=-1)
    next_token_logits = next_token_logits / args.temperature
    # 对于<unk>的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
    next_token_logits[unk_id] = -float('Inf')
    state = instance['input_ids'].squeeze(0)[query_length:]

    # TODO 这个是预防2-gram的重复的，要检查是否有特殊的Token
    if len(state) > 0:
        last_id = state[-1]
        for idx in range(0, len(state) - 1):
            if last_id == state[idx]:
                pid = state[idx + 1]
                next_token_logits[pid] = next_token_logits[pid] - abs(next_token_logits[pid]) * repetition_penalty

    if target_length > 0:
        target_length_penalty = max(0, (target_length - len(state)) / target_length) * ending_penalty
        next_token_logits[eos_id] = next_token_logits[eos_id] - abs(next_token_logits[eos_id]) * target_length_penalty

    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
    # torch.multinomial表示从候选集合中选出无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
    next_token_probs = F.softmax(filtered_logits, dim=-1)

    if not beam_mode:
        next_token_id = torch.topk(next_token_probs, 1)[1]
        return past_key_values, next_token_id, torch.log(raw_probs[next_token_id]).item()
    else:
        assert num_samples <= args.topk, 'num_samples should be less than or equal to topk '
        next_token_ids = torch.multinomial(next_token_probs, num_samples=num_samples)
        next_token_id_items = [x.item() for x in next_token_ids]
        next_logits = [torch.log(raw_probs[x]).item() for x in next_token_id_items]
        return None, next_token_ids, next_logits


def generate_next_token_beam(args, model, instance, unk_id, eos_id, query_length, repetition_penalty,
                             ending_penalty, target_length, num_samples, beam_with=1):
    """
    对于给定的上文，生成下一个单词
    """
    outputs = model.forward(instance['input_ids'],
                            segments=instance['segments'],
                            sources=instance['sources'],
                            source_index=instance['source_index'],
                            token_types=instance['token_types'],
                            pos_fw=instance['pos_fw'],
                            pos_bw=instance['pos_bw'],
                            word_aligns=instance['word_aligns'],
                            word_aligns_fw=instance['word_aligns_fw'],
                            word_aligns_bw=instance['word_aligns_bw'],
                            attention_mask=None,
                            labels=None,
                            past_key_values=instance.get('past_key_values', None),
                            )
    logits = outputs.logits
    past_key_values = outputs.past_key_values
    # next_token_logits表示最后一个token的hidden_state对应的prediction_scores,也就是模型要预测的下一个token的概率
    next_token_logits = logits[:, -1, :]
    raw_probs = F.softmax(next_token_logits, dim=-1)
    if args.temperature != 1:
        next_token_logits = next_token_logits / args.temperature
    # 对于<unk>的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
    next_token_logits[:, unk_id] = -1e20
    # </s> 也需要屏蔽
    next_token_logits[:, 2] = -1e20
    # 到这里
    beam_tokens = []
    beam_tokens_items = []
    beam_logits = []
    for beam_id in range(instance['input_ids'].size()[0]):
        state = instance['input_ids'][beam_id, query_length:]
        if len(state) > 0:
            last_id = state[-1]
            for idx in range(0, len(state) - 1):
                if last_id == state[idx]:
                    pid = state[idx + 1]
                    next_token_logits[beam_id, pid] = next_token_logits[beam_id, pid] - abs(
                        next_token_logits[beam_id, pid]) * repetition_penalty

        if target_length > 0:
            target_length_penalty = max(0, (target_length - len(state)) / target_length) * ending_penalty
            next_token_logits[beam_id, eos_id] = next_token_logits[beam_id, eos_id] - \
                                                 abs(next_token_logits[beam_id, eos_id]) * target_length_penalty

        filtered_logits = top_k_top_p_filtering(next_token_logits[beam_id], top_k=args.topk, top_p=args.topp,
                                                filter_value=-1e20)
        next_token_probs = F.softmax(filtered_logits, dim=-1)
        next_token_ids = torch.multinomial(next_token_probs, num_samples=num_samples)
        next_token_id_items = [x.item() for x in next_token_ids]
        next_logits = [torch.log(raw_probs[beam_id][x]).item() for x in next_token_id_items]
        beam_logits.append(next_logits)
        beam_tokens.append(next_token_ids)
        beam_tokens_items.append(next_token_id_items)
    return None, beam_tokens, beam_tokens_items, beam_logits


def update_new_token(args, tokenizer, past_key_values, instance, next_token_id
                     , next_source, next_source_index, next_type, next_internal_bw, next_word_aligns_bw,
                     device):
    next_token = tokenizer.decode(next_token_id)
    if 'aligns' not in instance:
        instance['aligns'] = 1
        instance['aligns_fw'] = 1
    else:
        if len(next_token) > 0 and next_token[0] == '▁':
            instance['aligns'] = instance['aligns'] + 1
            instance['aligns_fw'] = 1
        else:
            instance['aligns'] = instance['aligns']
            instance['aligns_fw'] = instance['aligns_fw'] + 1
    aligns, aligns_fw = [instance['aligns']], [instance['aligns_fw']]

    # 判断当前阶段
    aligns = torch.tensor(aligns, dtype=torch.long).to(device)
    aligns_fw = torch.tensor(aligns_fw, dtype=torch.long).to(device)

    instance['input_ids'] = torch.cat((instance['input_ids'], next_token_id.unsqueeze(0)), dim=1)
    instance['segments'] = torch.cat((instance['segments'], instance['segments'][:, -1:]), dim=1)
    instance['sources'] = torch.cat((instance['sources'], next_source), dim=1)
    instance['source_index'] = torch.cat((instance['source_index'], next_source_index), dim=1)
    instance['token_types'] = torch.cat((instance['token_types'], next_type.unsqueeze(0)), dim=1)
    instance['pos_fw'] = torch.cat((instance['pos_fw'], instance['pos_fw'][:, -1:] + 1), dim=1)
    instance['pos_bw'] = torch.cat((instance['pos_bw'], next_internal_bw), dim=1)
    instance['word_aligns'] = torch.cat((instance['word_aligns'], aligns.unsqueeze(0)), dim=1)
    instance['word_aligns_fw'] = torch.cat((instance['word_aligns_fw'], aligns_fw.unsqueeze(0)), dim=1)
    instance['word_aligns_bw'] = torch.cat((instance['word_aligns_bw'], next_word_aligns_bw), dim=1)
    return instance


def update_new_token_beams(tokenizer, merged_instances, merged_algins, next_token_id_items
                           ,merge_generation_modes, label_dict, device):
    last_token_items = [tokenizer.decode(x.item()) for x in merged_instances['input_ids'][:, -1]]
    next_token_items = [tokenizer.decode(x) for x in next_token_id_items]
    last_pos_fws = [x.item() for x in merged_instances['pos_fw'][:, -1]]

    aligns = []
    aligns_fw = []
    new_aligns = []
    num_repeats = len(merged_algins)
    next_generation_modes = []

    next_segment_ids = []
    next_source_ids = []
    next_source_indices = []
    next_types = []
    next_poses = []

    for beam_id, instance in enumerate(merged_algins):
        last_generation_mode = merge_generation_modes[beam_id]
        next_token = next_token_items[beam_id]
        last_token = last_token_items[beam_id]
        if last_generation_mode == 'knowledge':
            # 首先判断当前token是否结束的
            if next_token == label_dict['response_id']:
                next_source_indices.append([0])
                next_generation_modes.append('dialogue')
                next_types.append(label_dict['next_action_type'])
                instance = (0, 0)
                next_source_ids.append(label_dict['next_action_source'])
                next_poses.append([0])
            else:
                next_generation_modes.append('knowledge')
                next_source_ids.append(label_dict['next_csk_source'])
                next_source_indices.append([1])
                next_types.append(label_dict['next_csk_type'])
                if last_token in label_dict['special_csk_ids'] or next_token in label_dict['special_csk_ids']:
                    instance = (1, 1)
                else:
                    instance = (instance[0], instance[1] + 1)
                next_poses.append([last_pos_fws[beam_id] + 1])
        else:
            next_generation_modes.append('dialogue')
            next_source_ids.append(label_dict['next_source'])
            next_source_indices.append([1])
            next_types.append(label_dict['next_type'])
            next_poses.append([last_pos_fws[beam_id] + 1])
            if next_token == label_dict['m1r0_id']:
                instance = (0, 0)
            elif instance[0] is None or last_token == label_dict['m1r0_id']:
                instance = (1, 1)
            else:
                if len(next_token) > 0 and next_token[0] == '▁':
                    instance = (instance[0] + 1, instance[1])
                else:
                    instance = (instance[0], instance[1] + 1)
        aligns.append(instance[0])
        aligns_fw.append(instance[1])
        new_aligns.append((instance[0], instance[1]))
        if next_generation_modes[-1] == 'dialogue':
            next_segment_ids.append(label_dict['next_segment_id'])
        else:
            next_segment_ids.append(label_dict['next_csk_segment_id'])

    next_segment_ids = torch.cat(next_segment_ids, dim=0).to(device)
    next_source_ids = torch.cat(next_source_ids, dim=0).to(device)
    next_source_indices = torch.LongTensor(next_source_indices).to(device)
    next_poses = torch.LongTensor(next_poses).to(device)
    next_types = torch.cat(next_types).unsqueeze(-1).to(device)

    # 判断当前阶段
    aligns = torch.tensor(aligns, dtype=torch.long).to(device)
    aligns_fw = torch.tensor(aligns_fw, dtype=torch.long).to(device)
    next_token_ids = torch.tensor(next_token_id_items, dtype=torch.long).to(device)
    merged_instances['input_ids'] = torch.cat((merged_instances['input_ids'], next_token_ids.unsqueeze(1)), dim=1)
    merged_instances['segments'] = torch.cat((merged_instances['segments'], next_segment_ids), dim=1)
    merged_instances['sources'] = torch.cat((merged_instances['sources'], next_source_ids), dim=1)
    merged_instances['source_index'] = torch.cat((merged_instances['source_index'], next_source_indices), dim=1)
    merged_instances['token_types'] = torch.cat((merged_instances['token_types'], next_types), dim=1)
    merged_instances['pos_fw'] = torch.cat((merged_instances['pos_fw'], next_poses), dim=1)
    merged_instances['pos_bw'] = torch.cat(
        (merged_instances['pos_bw'], label_dict['next_internal_bw'].repeat(num_repeats, 1)), dim=1)
    merged_instances['word_aligns'] = torch.cat((merged_instances['word_aligns'], aligns.unsqueeze(1)), dim=1)
    merged_instances['word_aligns_fw'] = torch.cat((merged_instances['word_aligns_fw'], aligns_fw.unsqueeze(1)), dim=1)
    merged_instances['word_aligns_bw'] = torch.cat(
        (merged_instances['word_aligns_bw'], label_dict['next_word_aligns_bw'].repeat(num_repeats, 1)), dim=1)
    return merged_instances, new_aligns, next_generation_modes


def clone_beam(instance):
    new_instance = {}
    for k, v in instance.items():
        new_instance[k] = v
    return new_instance


def beam_generate(args, model, tokenizer, test_data, raw_tgt_len, device):
    next_segment_id = torch.tensor([[segment2id["[DResponse]"]]], dtype=torch.long)
    next_source = torch.tensor([[source2id["[Machine]"]]], dtype=torch.long)
    next_type = torch.tensor([type2id['word']], dtype=torch.long)

    next_csk_segment_id = torch.tensor([[segment2id["[gs_csk_knowledge]"]]], dtype=torch.long)
    next_csk_source = torch.tensor([[source2id["[GeneratedKnowledge]"]]], dtype=torch.long)
    next_csk_type = torch.tensor([type2id['gs_csk_knowledge']], dtype=torch.long)

    next_action_source = torch.tensor([[source2id["[Action]"]]], dtype=torch.long)
    next_action_type = torch.tensor([type2id['action']], dtype=torch.long)

    next_source_index = torch.tensor([[1]], dtype=torch.long)
    next_internal_bw = torch.tensor([[0]], dtype=torch.long)
    next_word_aligns_bw = torch.tensor([[0]], dtype=torch.long)

    next_type = next_type.to(device)
    next_segment_id = next_segment_id.to(device)
    next_source_index = next_source_index.to(device)
    next_source = next_source.to(device)
    next_internal_bw = next_internal_bw.to(device)
    next_word_aligns_bw = next_word_aligns_bw.to(device)
    next_csk_segment_id = next_csk_segment_id.to(device)
    next_csk_source = next_csk_source.to(device)
    next_csk_type = next_csk_type.to(device)
    next_action_source = next_action_source.to(device)
    next_action_type = next_action_type.to(device)

    if args.generation_mode == 'dialogue':
        init_dialogue_generation_mode = 'dialogue'
        eos_id = int(tokenizer.convert_tokens_to_ids("<sep>"))
    elif args.generation_mode == 'knowledge':
        init_dialogue_generation_mode = 'knowledge'
        eos_id = int(tokenizer.convert_tokens_to_ids("[DResponse]"))
    elif args.generation_mode == 'knowledge_dialogue':
        init_dialogue_generation_mode = 'knowledge'
        eos_id = int(tokenizer.convert_tokens_to_ids("<sep>"))
    else:
        raise NotImplementedError()

    label_dict = {
        'next_source_index': next_source_index,
        'next_internal_bw': next_internal_bw,
        'next_word_aligns_bw': next_word_aligns_bw,
        'next_type': next_type,
        'next_segment_id': next_segment_id,
        'next_source': next_source,
        'next_csk_segment_id': next_csk_segment_id,
        'next_csk_source': next_csk_source,
        'next_csk_type': next_csk_type,
        'next_action_source': next_action_source,
        'next_action_type': next_action_type,
        'response_id': "[DResponse]",
        'm1r0_id': "[M1R0]",
        'm1r0_idx': int(tokenizer.convert_tokens_to_ids("[M1R0]")),
        'special_csk_ids': {
            "[CSK_SEP]", "[TE_SEP]"
        }
    }

    for relation in csk_relation_set:
        label_dict['special_csk_ids'].add(relation)

    # 对title与context进行tokenize

    input_ids, full_labels, segments, sources, source_index, token_types, \
    pos_fw, pos_bw, word_aligns, word_aligns_fw, word_aligns_bw, query_length = test_data
    query_length = query_length[-1]
    first_instance = {
        'input_ids': input_ids[:, 0:query_length].to(device),
        'segments': segments[:, 0:query_length].to(device),
        'sources': sources[:, 0:query_length].to(device),
        'source_index': source_index[:, 0:query_length].to(device),
        'token_types': token_types[:, 0:query_length].to(device),
        'pos_fw': pos_fw[:, 0:query_length].to(device),
        'pos_bw': pos_bw[:, 0:query_length].to(device),
        'word_aligns': word_aligns[:, 0:query_length].to(device),
        'word_aligns_fw': word_aligns_fw[:, 0:query_length].to(device),
        'word_aligns_bw': word_aligns_bw[:, 0:query_length].to(device),
        'generation_mode': init_dialogue_generation_mode,
    }

    cur_len = len(first_instance['input_ids'])
    max_len = cur_len + raw_tgt_len
    unk_id = tokenizer.unk_token_id

    decoding_step = 0

    current_logits = []
    current_beams = []
    current_finished_states = []
    current_ids = []

    while True:
        decoding_step += 1
        if decoding_step == 1:
            past_key_values, next_token_ids, next_token_logits = generate_next_token(args, model, first_instance,
                                                                                     unk_id=unk_id,
                                                                                     eos_id=eos_id,
                                                                                     query_length=query_length,
                                                                                     ending_penalty=args.ending_penalty,
                                                                                     target_length=args.target_length,
                                                                                     repetition_penalty=args.repetition_penalty,
                                                                                     num_samples=args.beam_width,
                                                                                     beam_mode=True)
            # 创造current_beams
            for idx in range(args.beam_width):
                new_instance = clone_beam(first_instance)
                current_beams.append(new_instance)

            # 更新当前的Beams

            if init_dialogue_generation_mode == 'dialogue':
                my_next_source = next_source
                my_next_type = next_type
            else:
                my_next_source = next_csk_source
                my_next_type = next_csk_type

            for idx, beam in enumerate(current_beams):
                next_token_id = next_token_ids[idx:idx + 1]
                beam = update_new_token(args, tokenizer, None, beam, next_token_id, my_next_source,
                                        next_source_index, my_next_type, next_internal_bw, next_word_aligns_bw, device)
                current_beams[idx] = beam
                current_finished_states.append(False)
                current_logits.append([next_token_logits[idx]])
                current_ids.append([next_token_id.item()])
        else:
            flatten_next_token_id_items = []
            flatten_next_token_logits = []
            # 所有Beam进行解码
            # merge beams
            merge_instance = defaultdict(list)
            for is_finished, beam in zip(current_finished_states, current_beams):
                if is_finished:
                    continue
                for k, v in beam.items():
                    if isinstance(v, torch.Tensor):
                        merge_instance[k].append(v)
            for k in merge_instance.keys():
                merge_instance[k] = torch.cat(merge_instance[k], 0)

            beam_past_key_values, beam_token_ids, beam_tokens_items, beam_token_logits = generate_next_token_beam(args,
                                                                                                                  model,
                                                                                                                  merge_instance,
                                                                                                                  unk_id=unk_id,
                                                                                                                  eos_id=eos_id,
                                                                                                                  query_length=query_length,
                                                                                                                  ending_penalty=args.ending_penalty,
                                                                                                                  target_length=args.target_length,
                                                                                                                  repetition_penalty=args.repetition_penalty,
                                                                                                                  num_samples=args.beam_width,
                                                                                                                  beam_with=args.beam_width)
            if args.diverse_decoding > 0:
               for bid in range(len(beam_token_logits)):
                   for idx in range(len(beam_token_logits[bid])):
                       beam_token_logits[bid][idx] -= args.diverse_decoding * idx
            # Batch 只计算了当前没有Finished的部分
            valid_beam_id = 0
            for beam_id, beam in enumerate(current_beams):
                if not current_finished_states[beam_id]:
                    my_token_ids = beam_tokens_items[valid_beam_id]
                    my_token_logits = beam_token_logits[valid_beam_id]
                    valid_beam_id += 1
                else:
                    my_token_logits = [0] + [-float('Inf')] * (args.beam_width - 1)
                    my_token_ids = [eos_id] * args.beam_width
                flatten_next_token_id_items += my_token_ids
                flatten_next_token_logits += my_token_logits
            flatten_next_token_ids = torch.LongTensor(flatten_next_token_id_items).to(device)

            # 开始进行排序
            next_scores = []
            if args.sep_beam_scoring:
                for next_beam_candidate_id in range(len(flatten_next_token_id_items)):
                    last_beam_id = next_beam_candidate_id // args.beam_width
                    # 获得当前Beam的所有分数 **
                    my_logit = flatten_next_token_logits[next_beam_candidate_id]
                    my_logits = current_logits[last_beam_id] + [my_logit]
                    my_ids = current_ids[last_beam_id] + [flatten_next_token_id_items[next_beam_candidate_id]]
                    # 手先寸照response_id
                    response_start_pos = 0
                    while response_start_pos < len(my_ids) and my_ids[response_start_pos] != int(tokenizer.convert_tokens_to_ids("[DResponse]")):
                        response_start_pos += 1

                    to_merge_logits = my_logits[0:response_start_pos]
                    mean_logit = sum(to_merge_logits) / len(to_merge_logits)
                    my_logits = [mean_logit] + my_logits[response_start_pos:]
                    my_length = sum([int(x != 0) for x in my_logits])
                    # TODO 第二种长度惩罚工具
                    # （5+my_length)/my_length
                    my_score = sum(my_logits) / (my_length ** args.length_penalty)
                    next_scores.append((next_beam_candidate_id, my_score))
            else:
                # 计算每个Subbeam的分数
                previous_logits_sum = []
                previous_length = []
                for last_beam_id in range(args.beam_width):
                    # 预先计算Previous的分数
                    previous_logits_sum.append(sum(current_logits[last_beam_id]))
                    previous_length.append(sum([int(x != 0) for x in current_logits[last_beam_id]]))

                for next_beam_candidate_id in range(len(flatten_next_token_id_items)):
                    last_beam_id = next_beam_candidate_id // args.beam_width
                    # 获得当前Beam的所有分数 **
                    my_logit = flatten_next_token_logits[next_beam_candidate_id]
                    my_logits = previous_logits_sum[last_beam_id] + my_logit
                    my_length = previous_length[last_beam_id] + int(my_logit != 0)
                    # TODO 第二种长度惩罚工具
                    # （5+my_length)/my_length
                    my_score = my_logits / (my_length ** args.length_penalty)
                    next_scores.append((next_beam_candidate_id, my_score))

            # 开始升序排列，higher is better
            next_scores = sorted(next_scores, key=lambda x: x[1], reverse=True)
            # current_beams,current_lengths,current_ids,current_finished_states,current_logits
            next_beams = []
            next_ids = []
            next_finished_states = []
            next_logits = []

            update_flags = []
            for next_beam_candidate_id, score in next_scores[0:args.beam_width]:
                last_beam_id = next_beam_candidate_id // args.beam_width
                next_token_id = flatten_next_token_ids[next_beam_candidate_id].unsqueeze(0)
                next_token_id_item = next_token_id.item()
                if next_token_id_item != eos_id:
                    next_beam = clone_beam(current_beams[last_beam_id])
                    update_flags.append(True)
                else:
                    update_flags.append(False)
                    next_beam = current_beams[last_beam_id]
                next_logits.append(
                        current_logits[last_beam_id] + [flatten_next_token_logits[next_beam_candidate_id]])
                next_ids.append(current_ids[last_beam_id] + [next_token_id_item])
                next_finished_states.append(
                    next_token_id_item == eos_id or current_finished_states[last_beam_id])
                next_beams.append(next_beam)

            # UpdateBeams
            ids_to_update = []
            aligns_to_update = []
            merge_instance = defaultdict(list)
            merge_generation_modes = []
            for should_be_update, beam, next_token_id_item in zip(update_flags, next_beams, next_ids):
                if not should_be_update:
                    continue
                merge_generation_modes.append(beam['generation_mode'])
                for k, v in beam.items():
                    if isinstance(v, torch.Tensor):
                        merge_instance[k].append(v)
                ids_to_update.append(next_token_id_item[-1])
                aligns_to_update.append((beam.get('aligns', None), beam.get('aligns_fw', None)))
            for k in merge_instance.keys():
                merge_instance[k] = torch.cat(merge_instance[k], 0)

            if len(aligns_to_update) > 0:
                merge_instance, merge_aligns, next_generation_modes = update_new_token_beams(tokenizer, merge_instance, aligns_to_update,
                                                                      ids_to_update
                                                                      , merge_generation_modes, label_dict, device)
                valid_beam_idx = 0
                for beam_id, should_be_update in enumerate(update_flags):
                    beam = next_beams[beam_id]
                    if not should_be_update:
                        continue
                    for k, v in beam.items():
                        if isinstance(v, torch.Tensor):
                            beam[k] = merge_instance[k][valid_beam_idx:valid_beam_idx + 1]
                    beam['aligns'] = merge_aligns[valid_beam_idx][0]
                    beam['aligns_fw'] = merge_aligns[valid_beam_idx][1]
                    beam['generation_mode'] = next_generation_modes[valid_beam_idx]
                    valid_beam_idx += 1


            current_beams = next_beams
            current_logits = next_logits
            current_ids = next_ids
            current_finished_states = next_finished_states

            if decoding_step >= max_len:
                break
            has_unfinished = False
            for finished in current_finished_states:
                if not finished:
                    has_unfinished = True
                    break
            if has_unfinished is False:
                break
    if not args.fast_decoding and False:
        context = tokenizer.decode(first_instance['input_ids'].squeeze(0)[0:query_length])
    else:
        context = "Generation"
    results = []
    for idx, result_ids in enumerate(current_ids):
        result = tokenizer.decode(result_ids)
        result = result.replace('<sep>', '').replace('\n', '').replace('[DResponse]', '')
        results.append(result)

    return context.replace('<sep>', '').replace('\n', ''), results, -1


def inference(args, part_id, part_num, device):
    os.environ["CUDA_VISIBLE_DEVICES"] = device  # 此处设置程序使用哪些显卡
    args.cuda = torch.cuda.is_available() and not args.no_cuda  # 当用户使用GPU,并且GPU可用时
    device = 'cuda:0' if args.cuda else 'cpu'
    # device = 'cpu'
    # 创建日志对象
    logger = set_logger(args.log_path)

    # 初始化tokenizer
    tokenizer = get_tokenizer()

    # 创建模型
    if args.gpt2_hybrid:
        my_model = GPT2HybridLMHeadModel
    else:
        my_model = GPT2LMHeadModel

    model_config = GPT2Config.from_json_file(args.model_config)
    gpt_model = my_model(config=model_config)
    model = FineUniGPT(gpt_model=gpt_model, init_range=0.05)

    if args.checkpoint:  # 加载预训练模型
        logger.info('load checkpoint from {}'.format(args.checkpoint))
        model_path = args.checkpoint
        files = os.listdir(model_path)
        files = sorted(files, reverse=False)
        flag = False
        for file in files:
            if file[-3:] == '.pt':
                model_name = '%s/%s' % (model_path, file)
                if device == 'cpu':
                    checkpoint = torch.load(model_name, map_location=torch.device(device))
                else:
                    checkpoint = torch.load(model_name)
                model.load_state_dict(checkpoint['model'])
                logger.info('[CHECKPOINT] Loaded params from  :%s' % model_name)
                flag = True
                break
        if not flag:
            logger.info('[CHECKPOINT] No checkpoint is found in :%s' % model_path)
            raise FileNotFoundError()

    model.eval()
    model = model.to(device)
    test_dataset = load_dataset(logger, args, is_train=False, data_path=args.test_path,
                                part_id=part_id, part_num=part_num, max_line=args.max_inference_line)

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True
    )

    # 开始生成
    results = []
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, test_data in enumerate(test_dataloader):
            assert args.beam_width > 0
            context, result, score = beam_generate(args, model, tokenizer, test_data, args.raw_tgt_len, device)
            results += result
            diff_time = (time.time() - start_time) / (batch_idx + 1)
            est_time = diff_time * (len(test_dataloader) - batch_idx) / 3600
            print('%s/%s-%.2fH/' % (batch_idx, len(test_dataloader), est_time))
            print(context)
            print(result)
            print('---------')
    return results


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpt2_hybrid', action='store_true', help='启用双头GPT2，以加强生成')
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--generation_mode', default='dialogue', type=str, required=False, help='dialogue')
    parser.add_argument('--sep_beam_scoring', action='store_true', help='打开后，当进入到对话生成时将会重新计算分数')
    parser.add_argument('--max_inference_line', default=-1, type=int, required=False, help='最大的解码长度')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--beam_width', default=0, type=int, required=False, help='最高几选一')
    parser.add_argument('--topk', default=0, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--diverse_decoding', default=0.00, type=float, required=False, help='length_penalty')
    parser.add_argument('--length_penalty', default=1.00, type=float, required=False, help='length_penalty')
    parser.add_argument('--ending_penalty', default=0.0, type=float, required=False, help='减少Ending出现的概率')
    parser.add_argument('--target_length', default=0, type=int, required=False, help='目标长度')
    parser.add_argument('--repetition_penalty', default=0.5, type=float, required=False, help='重复惩罚参数')
    parser.add_argument('--context_len', default=200, type=int, required=False, help='每一步生成时，参考的上文的长度')
    parser.add_argument('--raw_tgt_len', default=30, type=int, required=False, help='生成的最长长度')
    parser.add_argument('--max_len', default=1020, type=int, required=False, help='生成的最长长度')
    parser.add_argument('--num_workers', default=0, type=int, required=False, help='生成的最长长度')
    parser.add_argument('--log_path', default='log/generate.log', type=str, required=False, help='日志存放位置')
    parser.add_argument('--model_type', default='dialog', type=str, required=False, help='模型对话')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--fast_decoding', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--checkpoint', type=str, default='model/epoch1', help='模型存放位置')
    parser.add_argument('--context', type=str, default='对话上下文', help='对话上下文')
    parser.add_argument('--test_file', type=str, default=None, help='对话上下文')
    parser.add_argument('--save_path', type=str, default=None, help='对话上下文')
    parser.add_argument('--test_path', type=str, default=None, help='对话上下文')
    parser.add_argument('--model_config', default='config/FineGPT_LC.json', type=str, required=False,
                        help='需要从头训练一个模型时，模型参数的配置文件')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='测试的batch size')
    parser.add_argument('--thread', default=5, type=int, required=False, help='测试的batch size')

    args = parser.parse_args()

    pool = Pool(args.thread)
    tasks = []

    raw_gpus = args.device.split(',')
    for idx in range(args.thread):
        # inference(args, idx, args.thread)
        task = pool.apply_async(inference, (args, idx, args.thread, raw_gpus[idx % len(raw_gpus)]))
        tasks.append(task)
    pool.close()
    pool.join()

    results = []
    for task in tasks:
        results += task.get()

    if args.beam_width == 0:
        interval = 1
    else:
        interval = args.beam_width

    # 第一个
    with open(args.save_path, 'w+') as fout:
        for line in results[::interval]:
            fout.write(line + '\n')

    with open(args.save_path + '.beam', 'w+') as fout:
        for line in results:
            fout.write(line + '\n')
