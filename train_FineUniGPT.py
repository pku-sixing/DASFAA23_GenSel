import argparse
import math
import os
import pickle
import random
import time
from datetime import datetime

import numpy
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import transformers
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
from GPTModels import GPT2HybridLMHeadModel


import model_helper
from FineUniGPT import FineUniGPT
from data_parallel import BalancedDataParallel
from dataset import FineUniGPTDataset
from utils import set_logger, set_random_seed

# Batch_size_steps
from vocab.special_vocab import type2id, get_tokenizer, segment2id

batch_size_scale = {}
for i in range(0, 16):
    batch_size_scale[i] = 1
for i in range(16, 40):
    batch_size_scale[i] = 1
for i in range(40, 80):
    batch_size_scale[i] = 1
for i in range(80, 150):
    batch_size_scale[i] = 1
for i in range(150, 300):
    batch_size_scale[i] = 1
for i in range(300, 513):
    batch_size_scale[i] = 2
for i in range(513, 768):
    batch_size_scale[i] = 3
for i in range(768, 1000):
    batch_size_scale[i] = 4
for i in range(1000, 1200):
    batch_size_scale[i] = 5


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden_source_attention_mask_rate', default=0.0, type=float, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--fullkg_source_attention_mask_rate', default=0.0, type=float, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--partialkg_source_attention_mask_rate', default=0.0, type=float, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--one_mask_type_one_time', action='store_true', help='启用双头GPT2，以加强生成')
    parser.add_argument('--gpt2_hybrid', action='store_true', help='启用双头GPT2，以加强生成')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--generative_teach_force_rate', default=0.0, type=float, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--generative_mask', action='store_true', help='是否进行Generative的Mask')
    parser.add_argument('--generative_mask_rate', default=0.15, type=float, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--generative_mask_mode', default='std', type=str, required=False,
                        help='需要从头训练一个模型时，模型参数的配置文件')
    parser.add_argument('--vocab_path', default='vocab/chinese_vocab.model', type=str, required=False,
                        help='sp模型路径')
    parser.add_argument('--model_config', default='config/FineGPT_LC.json', type=str, required=False,
                        help='需要从头训练一个模型时，模型参数的配置文件')
    parser.add_argument('--train_path', default='data/toy/train.pkl', type=str, required=False, help='经过预处理之后的数据存放路径')
    parser.add_argument('--dev_path', default='data/toy/train.pkl', type=str, required=False, help='经过预处理之后的数据存放路径')
    parser.add_argument('--max_len', default=1024, type=int, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--bin_number', default=0, type=int, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--log_path', default='log/FineGPT_LC.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--ignore_index', default=-100, type=int, required=False,
                        help='对于ignore_index的label token不计算梯度')
    parser.add_argument('--max_valid_vocab_size', default=30000, type=int, required=False, help='超过的部分vocab，不计算loss')
    parser.add_argument('--epochs', default=100, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--valid_per_epoch_num', default=1, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='训练的batch size')
    parser.add_argument('--start_epoch_num', default=1, type=int, required=False, help='开始训练的Epoch')
    # parser.add_argument('--batch_size', default=32, type=int, required=False, help='训练的batch size')
    parser.add_argument('--mem_size_offset', default=1, type=float, required=False,
                        help='当前预设定BatchSize的基准长度，后续将进行缩放')
    parser.add_argument('--gpu0_bsz', default=0.33, type=float, required=False, help='0号卡的batch size')
    parser.add_argument('--global_mask_rate', default=0.0, type=float, required=False, help='学习率')
    parser.add_argument('--sp_embed_init_range', default=0.02, type=float, required=False, help='学习率')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument('--log_step', default=100, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--sample_num', default=-1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--save_model_path', default='model', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--experiment_name', default='model', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--pretrained_model', default=None, type=str, required=False,
                        help='预训练的模型的路径')
    parser.add_argument('--checkpoint', default=None, type=str, required=False,
                        help='预训练的模型的路径')
    parser.add_argument('--seed', type=int, default=1234, help='设置随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up步数')
    # parser.add_argument('--label_smoothing', default=True, action='store_true', help='是否进行标签平滑')
    args = parser.parse_args()
    return args


def sequence_mask_fn(lengths, maxlen=None, dtype=torch.bool, mask_first=False):
    """
    :param lengths: [seq_len]
    :param maxlen:
    :param dtype:
    :return:
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device, requires_grad=False)
    matrix = torch.unsqueeze(lengths, dim=-1)
    # mask = row_vector < matrix
    mask = row_vector.lt(matrix)
    if mask_first:
        mask[:, 0:1] = False
    mask = mask.type(dtype)
    return mask


def collate_fn(batch):
    # input_ids, context_length, full_length
    raw_input_ids = [x[0] for x in batch]
    context_lengths = [x[1] for x in batch]
    full_lengths = [x[2] for x in batch]
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
    # full_lengths = torch.stack(full_lengths)
    # print(input_ids, labels, context_lengths, full_lengths)
    # padding labels by lengths
    # mask为True的地方是Context，需要屏蔽
    seq_mask = ~sequence_mask_fn(context_lengths, input_ids.size()[1])
    labels = torch.where(seq_mask, labels, torch.ones_like(labels) * -100)
    return input_ids, labels, segment_ids, source_ids, source_indices, type_ids, fw_pos, bw_pos, \
           word_align, word_index_fw, word_index_bw

def mask_collate_fn(batch, mask_id = None, generative_mask=True, generative_mask_rate=0.15, generative_mask_mode='std'):
    # input_ids, context_length, full_length
    raw_input_ids = [x[0] for x in batch]
    context_lengths = [x[1] for x in batch]
    full_lengths = [x[2] for x in batch]
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

    has_drop_mask = False
    if generative_mask:
        if generative_mask_mode == 'std':
            # 正常随机屏蔽
            is_valid_mask = torch.rand_like(input_ids.to(torch.float)) < generative_mask_rate
            generative_mask = segment_ids == segment2id["[gs_csk_knowledge]"]
            # 取交集后进行Masking
            masking = is_valid_mask & generative_mask
            has_drop_mask = True
            input_ids = torch.where(masking, torch.ones_like(input_ids) * mask_id, input_ids)
        elif generative_mask_mode == 'full_mask' or generative_mask_mode == 'full_mask_no_learn':
            if random.random() < generative_mask_rate:
                generative_mask = segment_ids == segment2id["[gs_csk_knowledge]"]
                input_ids = torch.where(generative_mask, torch.ones_like(input_ids) * mask_id, input_ids)
                if generative_mask_mode == 'full_mask_no_learn':
                    labels = torch.where(generative_mask, torch.ones_like(labels) * -100, labels)
        else:
            raise NotImplementedError()


    # mask为True的地方是Context，需要屏蔽
    seq_label_mask = ~sequence_mask_fn(context_lengths, input_ids.size()[1])


    labels = torch.where(seq_label_mask, labels, torch.ones_like(labels) * -100)
    return input_ids, labels, segment_ids, source_ids, source_indices, type_ids, fw_pos, bw_pos, \
           word_align, word_index_fw, word_index_bw


def load_dataset(logger, args, is_train=True,  epoch=1, data_path=None,
                 part_id=None, part_num=None, max_line=-1):
    """
    加载训练集
    如果是训练，则会随机打乱数据,只使用一部分数据
    """
    if data_path is None:
        if is_train:
            train_path = args.train_path
            if args.sample_num > -1:
                train_path = train_path.replace('BID', str((epoch) % args.sample_num ))
            logger.info("loading epoch {} training data from {}".format(epoch, train_path))
        else:
            logger.info("loading dev data")
            train_path = args.dev_path
    else:
        logger.info("loading  data from {}".format(data_path))
        train_path = data_path


    with open(train_path, "rb") as f:
        logger.info("loading data from %s" % train_path)
        train_list = pickle.load(f)
        if max_line > -1:
            train_list = train_list[0:max_line]

    if is_train:
        random.shuffle(train_list)
        if args.bin_number > 0:
            bin_number = args.bin_number
            logger.info("bin_number is %d" % args.bin_number)
            train_len_tuple = [(idx, len(x['input_ids'])) for idx, x in enumerate(train_list)]
            sorted_train_len_tuple = sorted(train_len_tuple, key=lambda x: x[1], reverse=False)
            print(sorted_train_len_tuple[0:3])
            print(sorted_train_len_tuple[-3:])
            # 根据Bin Number 进行分组
            bin_size = len(train_len_tuple) // bin_number + 1
            bin_order = []
            for bid in range(bin_number):
                # 当前组的数据
                bin_tuples = sorted_train_len_tuple[bid * bin_size: bid * bin_size + bin_size]
                if len(bin_tuples) == 0:
                    continue
                # 组内打乱
                if is_train:
                    random.shuffle(bin_tuples)
                logger.info("bin {}, first 3 {}".format(bid, bin_tuples[0:3]))
                # 获得打乱后的ID
                bin = [x[0] for x in bin_tuples]
                bin_order += bin
            bin_list = [train_list[x] for x in bin_order]
            train_list = bin_list

    if part_id is not None:
        total_num = len(train_list)
        part_num = total_num // part_num + 1
        print('my part {}, {}-{},{}'.format(part_id, part_id * part_num, part_id * part_num + part_num, total_num))
        train_list = train_list[part_id * part_num:part_id * part_num + part_num]

    train_dataset = FineUniGPTDataset(train_list, args.max_len)

    return train_dataset


def train_epoch(model, train_dataloader, optimizer, scheduler, logger,
                epoch, args, scaler):
    # TODO Infer的时候需要可能需要忽略SpecialToken带来的影响
    model.train()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    epoch_start_sec_time = time.time()

    total_loss = 0  # 记录下整个epoch的loss的总和
    epoch_correct_num = 0  # 每个epoch中,预测正确的word的数量
    epoch_total_num = 0  # 每个epoch中,预测的word的总数量


    for batch_idx, batch_data in enumerate(train_dataloader):
        full_input_ids, full_labels, full_segment_ids, full_source_ids, full_source_indices, full_token_types,\
        full_fw_pos, full_bw_pos, full_word_align, full_word_index_fw, full_word_index_bw = batch_data
        # 捕获cuda out of memory exception
        try:
            # 决定是否mask
            if args.global_mask_rate > 0.0:
                mask_pos = torch.rand_like(full_input_ids.to(torch.float)) < args.global_mask_rate
                invalid_mask = full_input_ids > 8
                mask_pos = mask_pos & invalid_mask
                masks = torch.ones_like(full_input_ids) * args.mask_id
                full_input_ids = torch.where(mask_pos, masks, full_input_ids)
            # 屏蔽一部分不能计算的special_token
            # mask = full_labels >= args.max_valid_vocab_size
            # full_labels = torch.where(mask, torch.ones_like(full_labels) * args.ignore_index, full_labels)
            # 当前的基准长度
            current_length = full_labels.size()[1]
            gradient_accumulation_steps = batch_size_scale[int(current_length * args.mem_size_offset)]
            # 确定当前每一片的大小(不能保证最后一个batch的大小，所以使用这个等效操作）
            slice_size = args.batch_size // gradient_accumulation_steps
            if args.batch_size // gradient_accumulation_steps != 0:
                slice_size += 1
            current_batch_size = full_labels.size()[0]
            # print(full_labels.size(), batch_size_scale[current_length])
            for start_index in range(0, args.batch_size, slice_size):
                if start_index >= current_batch_size:
                    continue
                input_ids = full_input_ids[start_index:start_index + slice_size].to(device)
                labels = full_labels[start_index:start_index + slice_size].to(device)
                segment_ids = full_segment_ids[start_index:start_index + slice_size].to(device)
                source_ids = full_source_ids[start_index:start_index + slice_size].to(device)
                source_indices = full_source_indices[start_index:start_index + slice_size].to(device)
                token_types = full_token_types[start_index:start_index + slice_size].to(device)
                fw_pos = full_fw_pos[start_index:start_index + slice_size].to(device)
                bw_pos = full_bw_pos[start_index:start_index + slice_size].to(device)
                word_align = full_word_align[start_index:start_index + slice_size].to(device)
                word_index_fw = full_word_index_fw[start_index:start_index + slice_size].to(device)
                word_index_bw = full_word_index_bw[start_index:start_index + slice_size].to(device)

                attention_mask = torch.ones_like(segment_ids).to(torch.bool)
                golden_source_attention_mask_rate = args.golden_source_attention_mask_rate
                fullkg_source_attention_mask_rate = args.fullkg_source_attention_mask_rate
                partialkg_source_attention_mask_rate = args.partialkg_source_attention_mask_rate

                if args.one_mask_type_one_time:
                    rates = {
                        'golden_source_attention_mask_rate': 0,
                        'fullkg_source_attention_mask_rate': 0,
                        'partialkg_source_attention_mask_rate': 0,
                    }
                    valid_set = []
                    if golden_source_attention_mask_rate > 0:
                        valid_set.append(('golden_source_attention_mask_rate', golden_source_attention_mask_rate))
                    if fullkg_source_attention_mask_rate > 0:
                        valid_set.append(('fullkg_source_attention_mask_rate', fullkg_source_attention_mask_rate))
                    if partialkg_source_attention_mask_rate > 0:
                        valid_set.append(('partialkg_source_attention_mask_rate', partialkg_source_attention_mask_rate))
                    current_one = random.sample(valid_set, 1)[0]
                    rates[current_one[0]] = current_one[1]
                    golden_source_attention_mask_rate = rates['golden_source_attention_mask_rate']
                    fullkg_source_attention_mask_rate = rates['fullkg_source_attention_mask_rate']
                    partialkg_source_attention_mask_rate = rates['partialkg_source_attention_mask_rate']

                if golden_source_attention_mask_rate > 0:
                    # mask 所有golden知识的部分，不学习如何生成，也不学习如何
                    current_batch_size = segment_ids.size()[0]
                    is_generative_knowledge_mask = segment_ids == segment2id["[gs_csk_knowledge]"]
                    is_valid_mask = torch.rand([current_batch_size, 1]) < golden_source_attention_mask_rate
                    is_valid_mask = is_valid_mask.to(is_generative_knowledge_mask.device)
                    my_attention_mask = ~ (is_generative_knowledge_mask & is_valid_mask)
                    labels = torch.where(my_attention_mask, labels, torch.ones_like(labels) * args.ignore_index)
                    input_ids = torch.where(my_attention_mask, input_ids, torch.ones_like(labels) * args.mask_id)
                    attention_mask = my_attention_mask & attention_mask
                if fullkg_source_attention_mask_rate > 0:
                    # mask 所有golden知识的部分，不学习如何生成，也不学习如何
                    current_batch_size = segment_ids.size()[0]
                    is_csk_knowledge_mask = segment_ids == segment2id["[csk_knowledge]"]
                    is_valid_mask = torch.rand([current_batch_size, 1]) < fullkg_source_attention_mask_rate
                    is_valid_mask = is_valid_mask.to(is_csk_knowledge_mask.device)
                    my_attention_mask = ~ (is_csk_knowledge_mask & is_valid_mask)
                    labels = torch.where(my_attention_mask, labels, torch.ones_like(labels) * args.ignore_index)
                    input_ids = torch.where(my_attention_mask, input_ids, torch.ones_like(labels) * args.mask_id)
                    attention_mask = my_attention_mask & attention_mask
                if partialkg_source_attention_mask_rate > 0:
                    current_batch_size = segment_ids.size()[0]
                    is_csk_knowledge_mask = segment_ids == segment2id["[csk_knowledge]"]
                    # [batch_size, sequence_length]
                    max_index = source_ids.max().item()
                    index_gates = numpy.random.rand(max_index+1) < partialkg_source_attention_mask_rate
                    is_valid_mask = torch.zeros_like(segment_ids).to(torch.bool)
                    for index in range(1, max_index+1):
                        if index_gates[index]:
                            is_current_index = source_ids == index
                            is_valid_mask = is_valid_mask | is_current_index
                    my_attention_mask = ~ (is_csk_knowledge_mask & is_valid_mask)
                    labels = torch.where(my_attention_mask, labels, torch.ones_like(labels) * args.ignore_index)
                    input_ids = torch.where(my_attention_mask, input_ids, torch.ones_like(labels) * args.mask_id)
                    attention_mask = my_attention_mask & attention_mask



                # print(attention_mask.to(torch.float32).mean())

                with autocast():
                    attention_mask = attention_mask.to(torch.long)
                    outputs = model.forward(input_ids, labels=labels, attention_mask=attention_mask,
                                            segments=segment_ids.to(device), sources=source_ids.to(device),
                                            source_index=source_indices.to(device),
                                            token_types=token_types.to(device), pos_fw=fw_pos.to(device),
                                            pos_bw=bw_pos.to(device), word_aligns=word_align.to(device),
                                            word_aligns_fw=word_index_fw.to(device),
                                            word_aligns_bw=word_index_bw.to(device),
                                            teach_force_mode=args.generative_teach_force_rate > 0,
                                            teach_force_rate=args.generative_teach_force_rate
                                            )
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        loss = outputs[1]
                    else:
                        logits = outputs.logits
                        loss = outputs.loss
                    loss = loss.mean()

                    # 统计该batch的预测token的正确数与总数
                    batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
                    # 统计该epoch的预测token的正确数与总数
                    epoch_correct_num += batch_correct_num
                    epoch_total_num += batch_total_num
                    # 计算该batch的accuracy
                    batch_acc = batch_correct_num / batch_total_num
                    my_batch_size = input_ids.size()[0]
                    total_loss += loss.item() * (my_batch_size/current_batch_size)
                    # TODO 可能需要加入最后一个Loss的校正
                    if gradient_accumulation_steps > 1:
                        loss = loss * (my_batch_size/current_batch_size)

                # Scales loss. 为了梯度放大.
                scaler.scale(loss).backward()

            # 进行一定step的梯度累计之后，更新参数
            if True:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # 更新参数
                scaler.step(optimizer)
                # 准备着，看是否要增大scaler
                scaler.update()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                step_time = (time.time() - epoch_start_sec_time) / (batch_idx + 1)
                logger.info(
                    "batch {}/{} of epoch {}, loss {}, batch_acc {}, lr {}, step time {}".format(
                        batch_idx + 1, len(train_dataloader), epoch + 1,
                        round(total_loss / (batch_idx + 1), 4),
                        round(batch_acc, 4), scheduler.get_lr(), round(step_time, 3)
                    ))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory, {}".format(labels.shape))
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                raise exception
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # save model
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss


def eval_epoch(epoch, model, dev_dataloader, logger, args):
    model.eval()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    epoch_start_sec_time = time.time()

    total_loss = 0  # 记录下整个epoch的loss的总和
    epoch_correct_num = 0  # 每个epoch中,预测正确的word的数量
    epoch_total_num = 0  # 每个epoch中,预测的word的总数量

    for batch_idx, batch_data in enumerate(dev_dataloader):
        # 捕获cuda out of memory exception
        full_input_ids, full_labels, full_segment_ids, full_source_ids, full_source_indices, full_token_types,\
        full_fw_pos, full_bw_pos, full_word_align, full_word_index_fw, full_word_index_bw = batch_data
        try:

            # 屏蔽一部分不能计算的special_token
            # mask = full_labels >= args.max_valid_vocab_size
            # full_labels = torch.where(mask, torch.ones_like(full_labels) * args.ignore_index, full_labels)

            # 当前的基准长度
            current_length = full_labels.size()[1]
            gradient_accumulation_steps = batch_size_scale[int(current_length * args.mem_size_offset)]
            # 确定当前每一片的大小(不能保证最后一个batch的大小，所以使用这个等效操作）
            slice_size = args.batch_size // gradient_accumulation_steps
            if args.batch_size // gradient_accumulation_steps != 0:
                slice_size += 1
            current_batch_size = full_labels.size()[0]
            for start_index in range(0, args.batch_size, slice_size):
                if start_index >= current_batch_size:
                    continue
                input_ids = full_input_ids[start_index:start_index + slice_size].to(device)
                labels = full_labels[start_index:start_index + slice_size].to(device)
                segment_ids = full_segment_ids[start_index:start_index + slice_size].to(device)
                source_ids = full_source_ids[start_index:start_index + slice_size].to(device)
                source_indices = full_source_indices[start_index:start_index + slice_size].to(device)
                token_types = full_token_types[start_index:start_index + slice_size].to(device)
                fw_pos = full_fw_pos[start_index:start_index + slice_size].to(device)
                bw_pos = full_bw_pos[start_index:start_index + slice_size].to(device)
                word_align = full_word_align[start_index:start_index + slice_size].to(device)
                word_index_fw = full_word_index_fw[start_index:start_index + slice_size].to(device)
                word_index_bw = full_word_index_bw[start_index:start_index + slice_size].to(device)
                outputs = model.forward(input_ids, labels=labels, attention_mask=None,
                                        segments=segment_ids.to(device), sources=source_ids.to(device),
                                        source_index=source_indices.to(device),
                                        token_types=token_types.to(device), pos_fw=fw_pos.to(device),
                                        pos_bw=bw_pos.to(device), word_aligns=word_align.to(device),
                                        word_aligns_fw=word_index_fw.to(device),
                                        word_aligns_bw=word_index_bw.to(device),
                                        teach_force_mode=args.generative_teach_force_rate > 0,
                                        teach_force_rate=1.0 # Eval模式下完全变成推理的
                                        )
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                # 统计该batch的预测token的正确数与总数
                batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
                # 统计该epoch的预测token的正确数与总数
                epoch_correct_num += batch_correct_num
                epoch_total_num += batch_total_num
                # 计算该batch的accuracy
                batch_acc = batch_correct_num / batch_total_num
                my_batch_size = input_ids.size()[0]
                total_loss += loss.item() * (my_batch_size / current_batch_size)

            if (batch_idx + 1) % args.log_step == 0:
                step_time = (time.time() - epoch_start_sec_time) / (batch_idx + 1)
                logger.info(
                    "valid of epoch {}, {}/{},loss {}, batch_acc {}, step time {}".format(
                        epoch + 1, (batch_idx + 1), len(dev_dataloader), round(loss.item(), 4),
                        round(batch_acc, 4), round(step_time, 3)
                    ))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory, {}".format(labels.shape))
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(dev_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    return epoch_mean_loss


def train(model, logger, args):
    if args.generative_mask:
        logger.info('generative_mask')
        my_collate_fn = lambda x: mask_collate_fn(x, mask_id=args.mask_id, generative_mask=True,
                                                  generative_mask_rate=args.generative_mask_rate,
                                                  generative_mask_mode=args.generative_mask_mode)
    else:
        my_collate_fn = collate_fn

    # 加载训练集和验证集
    dev_dataset = load_dataset(logger, args, is_train=False)
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=my_collate_fn,
        drop_last=True
    )
    train_dataset = load_dataset(logger, args, epoch=0)
    epoch_step = int(round(0.5 + len(train_dataset) / args.batch_size))
    passed_epoch = args.start_epoch_num - 1
    p_total = passed_epoch * epoch_step
    t_total = args.epochs * epoch_step - p_total
    logger.info('epoch step {}, passed_step {} remaining total_step: {}'.format(epoch_step, p_total, t_total))
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(0, args.warmup_steps - p_total), num_training_steps=t_total
    )

    logger.info('start training')

    train_losses = []  # 记录每个epoch的平均loss
    eval_losses = []

    scaler = GradScaler()
    # ========== start training ========== #
    for epoch in range(args.start_epoch_num - 1, args.epochs):
        # ========= 每一个Epoch都要加载数据========= #
        train_dataset = load_dataset(logger, args, epoch=epoch)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            collate_fn=my_collate_fn,
            drop_last=True
        )
        train_loss = 0
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args, scaler=scaler)
        train_losses.append(round(train_loss, 4))
        logger.info("train loss list:{}".format(train_losses))

        if epoch % args.valid_per_epoch_num == 0:
            with torch.no_grad():
                eval_loss = 0
                eval_loss = eval_epoch(
                    model=model, dev_dataloader=dev_dataloader,
                    logger=logger, epoch=epoch, args=args)
                eval_losses.append(round(eval_loss, 4))
                logger.info("eval_loss loss list:{}".format(eval_losses))
                logger.info('saving model for epoch {}'.format(epoch + 1))
                states = {
                    'eval_losses': eval_losses,
                    'train_losses': train_losses,
                    'epoch_num': epoch,
                }
                model_helper.save_model(logger, args.save_model_path, args.experiment_name, epoch, math.exp(eval_loss),
                                        model,
                                        optimizer,
                                        args, states, best_model=False)

                logger.info('best epoch saved')
                # 保留当前最相关的一个Epoch
                if min(eval_losses) == eval_losses[-1]:
                    logger.info("save the best valid epoch")
                    states = {
                        'eval_losses': eval_losses,
                        'train_losses': train_losses,
                        'epoch_num': epoch,
                    }
                    model_helper.save_model(logger, args.save_model_path, args.experiment_name, epoch,
                                            math.exp(eval_loss), model, optimizer,
                                            args, states, best_model=True)

                    logger.info('best epoch saved')

    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("eval_losses:{}".format(eval_losses))


def calculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # loss = F.cross_entropy(predict_logit, target, ignore_index=pad_idx)
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def main():
    # 初始化参数
    args = set_args()

    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.cuda = not args.no_cuda

    # 创建日志对象
    logger = set_logger(args.log_path)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:0' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))

    # 设置随机种子
    set_random_seed(args.seed, args.cuda)

    # 初始化tokenizer
    tokenizer = get_tokenizer()
    args.eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    args.pad_id = tokenizer.pad_token_id
    args.mask_id = tokenizer.convert_tokens_to_ids("<mask>")

    # 创建模型的输出目录
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 创建模型
    if args.gpt2_hybrid:
        my_model = GPT2HybridLMHeadModel
    else:
        my_model = GPT2LMHeadModel


    if args.pretrained_model:  # 加载预训练模型
        # assert my_model is GPT2LMHeadModel, '不能从checkpoint加载LM模型'
        logger.info('load parameter from {}'.format(args.pretrained_model))
        gpt_model = my_model.from_pretrained(args.pretrained_model)
        if my_model is GPT2HybridLMHeadModel:
            model_dict = gpt_model.state_dict()
            logger.info('[CHECKPOINT] clone lm_head to lm2_head')
            model_dict['lm2_head.weight'] = model_dict['lm_head.weight'].clone()
            gpt_model.load_state_dict(model_dict)
    else:  # 初始化模型
        logger.info('use fresh parameter')
        model_config = GPT2Config.from_json_file(args.model_config)
        gpt_model = my_model(config=model_config)


    model = FineUniGPT(gpt_model=gpt_model, init_range=args.sp_embed_init_range)

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

                # 进行参数检查，以适配GPT2HybridLMHeadModel
                if my_model is  GPT2HybridLMHeadModel:
                    if 'gpt_model.lm2_head.weight' not in checkpoint['model']:
                        logger.info('[CHECKPOINT] clone lm_head to lm2_head :%s' % model_name)
                        checkpoint['model']['gpt_model.lm2_head.weight'] = checkpoint['model']['gpt_model.lm_head.weight'].clone()

                model.load_state_dict(checkpoint['model'])



                logger.info('[CHECKPOINT] Loaded params from  :%s' % model_name)
                flag = True
                break
        if not flag:
            logger.info('[CHECKPOINT] No checkpoint is found in :%s' % model_path)
            raise FileNotFoundError()

    model = model.to(device)
    logger.info('model config:\n{}'.format(gpt_model.config.to_json_string()))
    logger.info('there are 200 additional_special_tokens')
    assert gpt_model.config.vocab_size == tokenizer.vocab_size + 200

    # 多卡并行训练模型
    if args.cuda and torch.cuda.device_count() > 1:
        # model = DataParallel(model).cuda()
        model = BalancedDataParallel(args.gpu0_bsz, model, dim=0).cuda()
        logger.info("use GPU {} to train".format(args.device))

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 记录参数设置
    logger.info("args:{}".format(args))

    train(model, logger, args)


if __name__ == '__main__':
    main()
