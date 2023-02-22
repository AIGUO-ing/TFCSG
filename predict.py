
import torch
from transformers import InputFeatures, BertTokenizer, BertConfig, BertModel
import numpy as np
PAD, CLS = '[PAD]', '[SEP]'  # padding符号, bert中综合信息符号

import pickle


def save_variable(v, filename):
    with open(filename, 'wb') as f: # 打开或创建名叫filename的文档。
        pickle.dump(v, f)  # 在文件filename中写入v
        f.close()  # 关闭文件，释放内存。
        return filename


def load_variavle(filename):
    try:
        with open(filename, 'rb+') as f:
            r = pickle.load(f)
            f.close()
            return r

    except EOFError:
        return

def load_dataset(content, pad_size=64):
    token = tokenizer.tokenize(content)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = tokenizer.convert_tokens_to_ids(token)
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    seq = []
    seq.append(seq_len)
    token_ids = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    # token_ids.resize_(32,32)
    # pad前的长度(超过pad_size的设为pad_size)
    seq_len = torch.Tensor(seq).unsqueeze(0).to(device)

    mask = torch.LongTensor(mask).unsqueeze(0).to(device)
    # mask.resize_(32, 32)
    return token_ids, seq_len, mask

def get_sentence_vec(model,input):
    tensor_input = load_dataset(input, 64)
    outputs = model(tensor_input[0],tensor_input[1],tensor_input[2])
    return outputs[1]


def get_cos_similar_multi(v1: list, v2: list):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./pretrained_model/bert-base-uncased')
    device = 'cuda'
    config = BertConfig.from_pretrained('./pretrained_model/bert-base-uncased')
    config.attention_probs_dropout_prob = 0.1  # 修改config的dropout系数
    config.hidden_dropout_prob = 0.1
    bert = BertModel.from_pretrained('./pretrained_model/bert-base-uncased', config=config)

    bert.load_state_dict(torch.load('./save_model/model'))
    bert.to(device)
    '''
    批量向量处理
    '''
    qe = open('data/generation/train_generation.txt', 'r', encoding='utf8')
    vec_768 = []
    for i, line in enumerate(qe.readlines(), start=0):
        line = line.split('$%&#')[0]
        vec_768.append((get_sentence_vec(bert,line.replace('\n', ''))[0].tolist()))
        print(i)
    save_variable(vec_768, 'data/vec_768_CSG_sof_unsup.vec')
    '''
    # 单个向量的召回
    # '''
    test_data = load_variavle('data/generation/test_CSG_stackoverflow')
    p_1 = 0
    p_10 = 0
    p_5 = 0
    map_10 = 0
    mrr = 0
    for i, data in enumerate(test_data):
        print(i)
        dict = {}
        sentence_text, target_index = data[0], data[1]
        vec_768 = load_variavle('data/vec_768_CSG_sof_unsup.vec')
        vec_one = get_sentence_vec(bert, sentence_text.replace('\n', ''))[0].tolist()
        sim = get_cos_similar_multi(vec_one, vec_768)[0].tolist()
        for i, score in enumerate(sim):
            dict[i] = score
        dict = sorted(dict.items(), key=lambda d: d[1], reverse=True)
        # ----------------- p@1的计算----------------
        if dict[0][0] in target_index:
            p_1 += 1
        # ------------------p@5的计算----------------
        temp_5 = 0
        for i in range(5):
            if dict[i][0] in target_index:
                temp_5 += 1
        p_5 += temp_5 / 5
        # ------------------p@10的计算---------------
        temp_10 = 0
        for i in range(10):
            if dict[i][0] in target_index:
                temp_10 += 1
        p_10 += temp_10 / 10
        # -----------------map@10的计算--------------
        cur = 0
        temp_map = 0
        for i in range(10):
            if dict[i][0] in target_index:
                cur = cur + 1
                temp_map += cur / (i + 1)
        if cur == 0:
            map_10 += 0
        else:
            map_10 += temp_map / cur
        # -----------------mrr的计算-----------------
        temp_mrr = 0
        for i in range(20):
            if dict[i][0] in target_index:
                temp_mrr += 1 / (i + 1)
                break
        mrr += temp_mrr
    print("P@1=", p_1 / len(test_data))
    print("P@5=", p_5 / len(test_data))
    print("P@10=", p_10 / len(test_data))
    print("MAP@10=", map_10 / len(test_data))
    print("MRR=", mrr / len(test_data))