# coding: utf-8
import random
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
    """
    如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码
    """
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
    读取文件数据，数据label和content以\t分隔
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
    根据训练集构建词表并存储
    :param train_dir: str，训练集路径
    :param vocab_dir: str，词表路径
    :param vocab_size: str，词表长度
    :return:
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
    with open_file(vocab_dir, "r") as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    # dict的key为word，value为index
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def build_category(train_dir, category_dir):
    """
    根据训练数据创建类别表并存储
    :param train_dir: str，训练集路径
    :param category_dir: str，类别表路径
    :return:
    """
    _, train_data_category = read_file(train_dir)
    labels = list(set(train_data_category))
    # 对类别表排序，保证类别ID一致
    labels = sorted(labels)
    open_file(category_dir, mode='w').write('\n'.join(labels) + '\n')


def read_category(category_dir):
    """
    读取类别表
    Returns:
        categories: 类别列表
        cat_to_id: key为类别，value为index的词典
    """
    with open_file(category_dir, "r") as fp:
        # 如果是py2 则每个值都转化为unicode
        categories = [native_content(_.strip()) for _ in fp.readlines()]
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """
    将wordId表示的文本转化为word表示
    :param content: str，以wordId表示的文本
    :param words: list，id_to_word
    :return:
    """
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """
    将word表示的文本转化为wordId表示，并进行padding
    :param filename: str，文件路径
    :param word_to_id: dict，key为词，value为index的词典
    :param cat_to_id: dict，key为类别，value为index的词典
    :param max_length: int，序列最大长度
    :returns
        x_pad: padding/truncate后的词序列列表
        y_pad: 转换为one-hot编码后的类别列表
    """
    contents, labels = read_file(filename)

    # 对数据shuffle
    idx = list(range(len(contents)))
    random.shuffle(idx)
    contents = [contents[i] for i in idx]
    labels = [labels[i] for i in idx]

    data_id, label_id = [], []
    for i in range(len(contents)):
        # word_to_id是个字典，key是word，value是id
        # 将正文中的字转化为字id来表示，如果字不在vocab中则用<UNK>替换
        data_id.append([word_to_id[x] if x in word_to_id else word_to_id["<UNK>"] for x in contents[i]])

        # 将类别转化为类别id
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度，以0来padding，所以字表中<PAD>的id需要为0
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding="post", truncating="post")
    # 将标签转换为one-hot向量
    y_pad = kr.utils.to_categorical(label_id)

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """
    生成batch数据
    :param x:
    :param y:
    :param batch_size:
    :return: batch迭代器
    """
    data_len = len(x)
    # 计算一个epoch有几个batch
    num_batch = int((data_len - 1) // batch_size) + 1

    # 进行数据shuffle
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = int(i * batch_size)
        end_id = int(min((i + 1) * batch_size, data_len))
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
