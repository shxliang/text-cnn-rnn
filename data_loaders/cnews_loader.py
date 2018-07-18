# coding: utf-8

import sys
from collections import Counter
from importlib import reload

import numpy as np
import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    """
    以UTF-8解码字符串
    :param content: 字符串
    :return: 解码后字符串
    """
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """
    读取cnews文件数据
    数据label和content以制表符分隔
    :param filename: 文件路径
    :return: 
        content: 存储正文的列表
        label: 储存类别的列表
    """
    contents, labels = [], []
    with open_file(filename, "r") as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except Exception as e:
                print(e)
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """
    根据训练集构建词汇表，存储
    """
    train_data_content, _ = read_file(train_dir)

    all_data_content = []
    for content in train_data_content:
        all_data_content.extend(content)

    # 按字切分计数
    counter = Counter(all_data_content)
    # 保留词频数前vocab_size的字
    count_pairs = counter.most_common(vocab_size - 1)

    # 将count_pairs中每个tuple的第一个元素合并成一个tuple
    words, _ = list(zip(*count_pairs))
    # 在列表头添加一个<PAD>，<PAD>用来将所有文本pad为同一长度，<UNK>代表集外字
    words = ["<PAD>", "<UNK>"] + list(words)
    # 以\n来连接words中的各元素
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """
    读取词汇表
    Returns:
        words: 读取的词表中的词
        word_to_id: key为词，value为index的词典
    """
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir, "r") as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    # dict的key为word，value为index
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """
    读取分类目录，固定
    Returns:
        categories: 类别列表
        cat_to_id: key为类别，value为index的词典
    """

    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    # 二分类，识别是否为时政
    # categories = ["其他", "时政"]

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """
    将文件转换为id表示
    :param filename: 文件路径
    :param word_to_id: key为词，value为index的词典
    :param cat_to_id: key为类别，value为index的词典
    :param max_length: 序列最大长度
    :returns
        x_pad: padding/truncate后的词序列列表
        y_pad: 转换为one-hot编码后的类别列表
    """
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        # word_to_id是个字典，key是word，value是id
        # 将正文中的字转化为字id来表示，如果字不在vocab中则用<UNK>替换
        data_id.append([word_to_id[x] if x in word_to_id else word_to_id["<UNK>"] for x in contents[i]])

        # 将类别转化为类别id
        label_id.append(cat_to_id[labels[i]])
        # 只分为时政和其他
        # label_id.append(cat_to_id[labels[i]] if labels[i] == "时政" else cat_to_id["其他"])

    # 使用keras提供的pad_sequences来将文本pad为固定长度，以0来padding，所以字表中<PAD>的id需要为0
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding="post", truncating="post")
    # 将标签转换为one-hot向量
    y_pad = kr.utils.to_categorical(label_id)

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """
    生成batch数据
    """
    data_len = len(x)
    # 计算一个epoch有几个batch
    num_batch = int((data_len - 1) / batch_size) + 1

    # 进行数据shuffle
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
