import argparse
from utils import set_logger
from transformers import CpmTokenizer
import os
import pickle
from tqdm import tqdm


def preprocess():
    """
    对故事数据集进行预处理
    """
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab/chinese_vocab.model', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--log_path', default='log/preprocess.log', type=str, required=False, help='日志存放位置')
    parser.add_argument('--data_type', default='dialog', type=str, required=False, help='数据模式，默认对话')
    parser.add_argument('--data_path', default='data/toy/raw', type=str, required=False, help='数据集存放位置')
    parser.add_argument('--save_path', default='data/toy/debug.pkl', type=str, required=False, help='对训练数据集进行tokenize之后的数据存放位置')
    parser.add_argument('--win_size', default=100, type=int, required=False, help='滑动窗口的大小，相当于每条数据的最大长度')
    parser.add_argument('--min_size', default=10, type=int, required=False, help='最小的数据长度')
    parser.add_argument('--step', default=70, type=int, required=False, help='滑动窗口的滑动步幅')
    args = parser.parse_args()

    # 初始化日志对象
    logger = set_logger(args.log_path)

    # 初始化tokenizer
    tokenizer = CpmTokenizer(vocab_file="vocab/chinese_vocab.model")
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")   # 文档结束符
    sep_id = tokenizer.sep_token_id

    # 读取所有对话内容
    train_list = []
    logger.info("start tokenizing data")
    for file in tqdm(os.listdir(args.data_path)):
        file = os.path.join(args.data_path, file)
        if args.data_type == 'dialog':
            if file.endswith('src') is False:
                logger.info("Skip {}".format(file))
                continue
            else:
                logger.info("Load ke_dialog {}".format(file))
                srcs = open(file, "r", encoding="utf8").readlines()
                tgts = open(file.replace('.src', '.tgt'), "r", encoding="utf8").readlines()
                for src, tgt in zip(srcs, tgts):
                    src = ''.join(src.strip().split())
                    src = tokenizer.encode(src, add_special_tokens=False)
                    tgt = ''.join(tgt.strip().split())
                    tgt = tokenizer.encode(tgt, add_special_tokens=False)

                    token_ids = src + [sep_id] + tgt + [eod_id]
                    win_size = args.win_size
                    start_index = max(0, len(token_ids) - win_size)
                    train_list.append(token_ids[start_index:])
        else:
            with open(file, "r", encoding="utf8")as reader:
                lines = reader.readlines()
                for article in lines:
                    article = ''.join(article.strip().split())
                    article_ids = tokenizer.encode(article, add_special_tokens=False)
                    # token_ids = title_ids + [sep_id] + article_ids + [eod_id]
                    token_ids = article_ids + [eod_id]
                    # train_list.append(token_ids)

                    # 对于每条数据，使用滑动窗口对其进行截断
                    win_size = args.win_size
                    step = args.step
                    start_index = 0
                    end_index = win_size
                    data = token_ids[start_index:end_index]
                    train_list.append(data)
                    start_index += step
                    end_index += step
                    while end_index+args.min_size < len(token_ids):  # 剩下的数据长度，大于或等于50，才加入训练数据集
                        data = token_ids[start_index:end_index]
                        train_list.append(data)
                        start_index += step
                        end_index += step

    # 序列化训练数据
    logger.info("total data num: %d" % len(train_list))
    with open(args.save_path, "wb") as f:
        pickle.dump(train_list, f)


if __name__ == '__main__':
    preprocess()


