import argparse
import json
import os
import pickle
from os.path import join
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from collections import namedtuple
import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
from transformers import InputFeatures, BertTokenizer, BertConfig, BertModel
import logging
from model.tokenizer import Tokenizer, load_base_vocab
from model.CSG import CSG, GSC_loss

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")


def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# contrasive data load
def load_train_data_contrasive(tokenizer, args):
    """
    获取无监督训练语料
    """
    logger.info('loading unsupervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-unsupervise.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            return feature_list
    feature_list = []
    with open(args.train_contrasive_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        # lines = lines[:100]
        logger.info("len of train data:{}".format(len(lines)))
        for line in tqdm(lines):
            line = line.strip()
            s1, s2 = str(line).split('$%&#')
            feature = tokenizer([s1, s2], max_length=args.max_len, truncation=True, padding='max_length',
                                return_tensors='pt')
            feature_list.append(feature)
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


# KL sentence and target data load
def load_train_data_KL(dir_path, tokenizer, args):
    """
    获取无监督训练语料
    """
    """
       读原始数据
       """
    feature_list = []
    with open(dir_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        # lines = lines[:100]
        logger.info("len of train data:{}".format(len(lines)))
        for line in tqdm(lines):
            line = line.strip()
            s1, s2 = str(line).split('$%&#')
            feature_s1 = tokenizer(s1, max_length=args.max_len, truncation=True, padding='max_length',
                                   return_tensors='pt')
            feature_s2 = tokenizer(s2, max_length=args.max_len, truncation=True, padding='max_length',
                                   return_tensors='pt')
            feature = (feature_s1,feature_s2)
            feature_list.append(feature)
    return feature_list


class DatasetGen(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, sents_src, sents_tgt, args):
        ## 一般init函数是加载所有数据
        super(DatasetGen, self).__init__()
        # 读原始数据
        self.sents_src, self.sents_tgt = sents_src, sents_tgt
        self.word2idx = load_base_vocab(args.pretrain_model_path + '/vocab.txt')
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids, token_type_ids = self.tokenizer.encode(src, tgt, max_length=args.max_len)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):
        return len(self.sents_src)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.LongTensor(pad_indice)

    def torch_unilm_mask(s):
        '''
        使用torch计算unilm mask
        :param s: input
        :return:
        用法
        torch.cumsum(input, dim, out=None) → Tensor

        返回输入沿指定维度的累积和。例如，如果输入是一个N元向量，则结果也是一个N元向量，第i 个输出元素值为 yi=x1+x2+x3+…+xi
        '''

        idxs = torch.cumsum(s, axis=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        return mask.float()

    token_ids = [data["token_ids"] for data in batch]
    max_length = args.max_len  # max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    token_ids_padded = padding(token_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()
    token_type_ids_padded = padding(token_type_ids, max_length)
    attention_mask = torch_unilm_mask(token_type_ids_padded)

    return token_ids_padded, token_type_ids_padded, target_ids_padded, attention_mask


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def train(model, train_dataloader_cl, train_dataloader_gen,train_dataloader_KL, tokenizer, optimizer, args):
    logger.info("start training")
    model.train()
    device = args.device
    for epoch in range(args.epochs):
        train_data = zip(train_dataloader_cl,  train_dataloader_gen, train_dataloader_KL)
        train_data = list(map(list, train_data))
        print('epoch={}'.format(epoch))
        for batch_idx, (data_cl, data_gen, data_kl) in enumerate(tqdm(train_data)):
            klsequence,kltarget = data_kl
            sq_len = klsequence['input_ids'].shape[-1]
            S_seq_input_ids = klsequence['input_ids'].view(-1, sq_len).to(device)
            S_seq_attention_mask = klsequence['attention_mask'].view(-1, sq_len).to(device)
            S_seq_token_type_ids = klsequence['token_type_ids'].view(-1, sq_len).to(device)
            S_tgt_input_ids = kltarget['input_ids'].view(-1, sq_len).to(device)
            S_tgt_attention_mask = kltarget['attention_mask'].view(-1, sq_len).to(device)
            S_tgt_token_type_ids = kltarget['token_type_ids'].view(-1, sq_len).to(device)
            sql_len = data_cl['input_ids'].shape[-1]
            C_input_ids = data_cl['input_ids'].view(-1, sql_len).to(device)
            C_attention_mask = data_cl['attention_mask'].view(-1, sql_len).to(device)
            C_token_type_ids = data_cl['token_type_ids'].view(-1, sql_len).to(device)
            G_token_ids, G_token_type_ids, G_target_ids, G_att_mask = data_gen
            G_token_ids = G_token_ids.to(device)
            G_token_type_ids = G_token_type_ids.to(device)
            G_att_mask = G_att_mask.to(device)
            G_target_ids = G_target_ids.to(device)
            KL_loss, out_cl, G_loss = model(
                S_seq_input_ids=S_seq_input_ids,
                S_seq_attention_mask=S_seq_attention_mask,
                S_seq_token_type_ids=S_seq_token_type_ids,
                S_tgt_input_ids=S_tgt_input_ids,
                S_tgt_attention_mask=S_tgt_attention_mask,
                S_tgt_token_type_ids=S_tgt_token_type_ids,
                C_input_ids=C_input_ids,
                C_attention_mask=C_attention_mask,
                C_token_type_ids=C_token_type_ids,
                G_input_tensor=G_token_ids,
                G_token_type_id=G_token_type_ids,
                G_attention_mask=G_att_mask,
                G_labels=G_target_ids
            )

            loss = GSC_loss(KL_loss, out_cl.pooler_output, G_loss, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def read_corpus_gen(dir_path):
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []
    with open(dir_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        logger.info("len of train data:{}".format(len(lines)))
        for line in tqdm(lines):
            line = line.strip()
            s1, s2 = str(line).split('$%&#')
            sents_src.append(s1)
            sents_tgt.append(s2)
    return sents_src, sents_tgt


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"], \
        'pooler should in ["cls", "pooler", "last-avg", "first-last-avg"]'

    config = BertConfig.from_pretrained(args.pretrain_model_path)
    config.attention_probs_dropout_prob = args.dropout  # 修改config的dropout系数
    config.hidden_dropout_prob = args.dropout
    plm = BertModel.from_pretrained(args.pretrain_model_path, config=config)  # bert模型,预训练结束后保存的训练模型plm

    model = CSG(plm, config, args=args).to(
        args.device)
    if args.do_train:
        assert args.train_mode in ['supervise', 'unsupervise'], \
            "train_mode should in ['supervise', 'unsupervise']"
        train_data_s = load_train_data_KL(args.train_generation_file, tokenizer, args)
        train_dataset_s = TrainDataset(train_data_s, tokenizer, max_len=args.max_len)
        train_dataloader_KL = DataLoader(train_dataset_s, batch_size=args.batch_size_train, shuffle=False,
                                         num_workers=args.num_workers)
        train_data_cl = load_train_data_contrasive(tokenizer, args)
        train_dataset_cl = TrainDataset(train_data_cl, tokenizer, max_len=args.max_len)
        train_dataloader_cl = DataLoader(train_dataset_cl, batch_size=args.batch_size_train, shuffle=False,
                                         num_workers=args.num_workers)
        sents_src, sents_tgt = read_corpus_gen(args.train_generation_file)
        train_dataset_gen = DatasetGen(sents_src, sents_tgt, args)
        train_dataloader_gen = DataLoader(train_dataset_gen, batch_size=args.batch_size_train, shuffle=False,
                                          collate_fn=collate_fn, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train(model, train_dataloader_cl, train_dataloader_gen,train_dataloader_KL, tokenizer, optimizer, args)
    logging.info(" Saving fine-tuned model")
    torch.save(plm.state_dict(), args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument("--output_path", type=str, default='output')
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_step", type=int, default=100, help="every eval_step to evaluate model")
    parser.add_argument("--max_len", type=int, default=64, help="max length of input")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--train_generation_file", type=str, default="data/generation/train_sof.txt")
    parser.add_argument("--train_contrasive_file", type=str, default="data/generation/train_sof.txt")
    parser.add_argument("--pretrain_model_name", type=str,
                        default="bert-base-uncased")
    parser.add_argument("--pretrain_model_path", type=str,
                        default="pretrained_model/bert-base-uncased")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='cls', help='pooler to use')
    parser.add_argument("--train_mode", type=str, default='unsupervise', choices=['unsupervise', 'supervise'],
                        help="unsupervise or supervise")
    parser.add_argument("--overwrite_cache", action='store_true', default=False, help="overwrite cache")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_predict", action='store_true', default=True)
    parser.add_argument('--output_dir', type=Path, default='./save_model/model')
    parser.add_argument('--type', type=str, default='sentence', choices=['document', 'sentence'])
    parser.add_argument("--do_lower_case", action="store_true", default=True)
    parser.add_argument("--reduce_memory", action="store_true", default=False,
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    args = parser.parse_args()
    seed_everything(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")
    args.output_path = join(args.output_path, args.train_mode,
                            'bsz-{}-lr-{}-dropout-{}'.format(args.batch_size_train, args.lr, args.dropout))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    main(args)
