import collections
import pickle
import string
import gensim
import jieba
import jieba.posseg as psg
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from scipy.optimize import linear_sum_assignment

from GSDMM_filter.MovieGroupProcess import MovieGroupProcess

output_path = '../LDA_EN/result'
file_path = '../LDA_EN/data'
data = pd.read_excel('D:\PyCharm\PythonCode\QA_Retrieval\\unsupclass\LDA_EN\data_sof.xlsx')  # content type
dic_file = '../LDA_EN/dict.txt'
stop_file = '../LDA_EN/stopwords.txt'

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


import re


# a= 'bcshdc心塞爱%课程124/*'
def filter_str(desstr):
    restr = ' '
    # 过滤除中英文及数字以外的其他字符
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z]")
    return res.sub(restr, desstr)


# print(filter_str(a))
def english_clean(doc):
    doc = filter_str(doc)
    normalized = []
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    for word in punc_free.split():
        normalized.append(lemma.lemmatize(word))
    # normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def chinese_word_cut(mytext):
    jieba.load_userdict(dic_file)
    jieba.initialize()
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ['n', 'nz', 'vn']
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)

    word_list = []
    # jieba分词
    seg_list = psg.cut(mytext)
    for seg_word in seg_list:
        # word = re.sub(u'[^\u4e00-\u9fa5]','',seg_word.word)
        word = seg_word.word

        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word) < 2:  # this word is stopword
                find = 1
                break
        if find == 0 and seg_word.flag in flag_list:
            word_list.append(word)
    return word_list


content = data['content'].to_list()
data['content'] = data['content'].astype(str)
data["content_cutted"] = data.content.apply(english_clean)
docs = data["content_cutted"].to_numpy()
dictionary = gensim.corpora.Dictionary(docs)
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=2000)
vocab_length = len(dictionary)
bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
# 初始化GSDMM
gsdmm = MovieGroupProcess(K=20, alpha=0.5, beta=0.5, n_iters=20)
# 拟合GSDMM模型
y = gsdmm.fit(docs, vocab_length)
data['topic'] = y
label = data['type'].to_numpy()
predict = data['topic'].to_numpy()
print("Acc={}".format(acc(label, predict)))

data.to_excel("data_ch_topic.xlsx", index=False)
# 打印每个主题的文档数
doc_count = np.array(gsdmm.cluster_doc_count)
print('Number of documents per topic :', doc_count)

# 按分配给主题的文档数排序的主题
top_index = doc_count.argsort()[-20:][::-1]
print('Most important clusters (by number of docs inside):', top_index)

# 定义函数以获取每个主题的热门单词
topic_word_collection = []


def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts = sorted(cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        topic_word_collection.append(sort_dicts)
        print("\nCluster %s : %s" % (cluster, sort_dicts))


# 获取主题中的热门单词
top_words(gsdmm.cluster_word_distribution, top_index, 50)


# 将主题词的前200个存储到系统的变量中
def save_variable(v, filename):
    f = open(filename, 'wb')  # 打开或创建名叫filename的文档。
    pickle.dump(v, f)  # 在文件filename中写入v
    f.close()  # 关闭文件，释放内存。
    return filename


def load_variavle(filename):
    try:
        f = open(filename, 'rb+')
        r = pickle.load(f)
        f.close()
        return r

    except EOFError:
        return ""

save_variable(topic_word_collection, 'twc')
twc = load_variavle('twc')
docs = list(docs)


def generate_topic_word(docs, twc):
    docs_topic = []
    for data in docs:
        topic_keyword = []
        for word in data:
            for topic in twc:
                for tw in topic:
                    w, tf = tw
                    if word == w or w == word:
                        if w not in topic_keyword:
                            topic_keyword.append(w)
        docs_topic.append(topic_keyword)
    return docs_topic


generate_topic_word = generate_topic_word(docs, twc)

# 如果 generate_topic_word 中有空的时候，使用docs填补
for i, data in enumerate(generate_topic_word):
    if len(data) == 0:
        generate_topic_word[i] = docs[i]


# 定义存储函数
def save_to_csv(list1, list2):
    if not len(list1) == len(list2):
        return 'list1 length is not equal to list2 length'
    else:

        out_files = open('train_topic.csv', 'w', encoding='utf8')
        for i in range(len(list1)):
            out_files.write(list1[i] + '$%&#' + ' '.join(wd for wd in list2[i]) + '\n')
        out_files.close()
        return


save_to_csv(content, generate_topic_word)
