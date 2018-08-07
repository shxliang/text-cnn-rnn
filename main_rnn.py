#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import tensorflow as tf

from data_loaders.data_loader import read_vocab, read_category, build_vocab, build_category
from infers.evaluate_one import evaluate_one
from models.rnn_model import TextRNN
from trainers.cnn_trainer import train, test
from utils.config_utils import load_config, create_rnn_config_model
from utils.utils import clean, mkdir

flags = tf.app.flags
flags.DEFINE_boolean("clean", True, "Whether clean train folder")
flags.DEFINE_boolean("train", True, "Whether train the model")

# configurations for the model
flags.DEFINE_integer("embedding_dim", 64, "词向量维度")
flags.DEFINE_integer("seq_length", 200, "序列长度")
flags.DEFINE_integer("num_classes", 4, "类别数")
flags.DEFINE_integer("vocab_size", 10000, "词汇表大小")
flags.DEFINE_integer("num_layers", 2, "隐藏层层数")
flags.DEFINE_integer("hidden_dim", 128, "全连接层神经元")
flags.DEFINE_string("rnn", "gru", "RNN Cell 类型")

# configurations for training
flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout保留比例")
flags.DEFINE_float("learning_rate", 0.001, "学习率")
flags.DEFINE_float("batch_size", 64, "每批训练大小")
flags.DEFINE_integer("num_epochs", 100, "总迭代轮次")
flags.DEFINE_integer("print_per_batch", 5, "每多少轮输出一次结果")
flags.DEFINE_integer("save_per_batch", 10, "每多少轮存入tensorboard")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")

flags.DEFINE_string("tensorboard_dir", os.path.join("tensorboard", "textcnn"), "TensorBoard Direction")
flags.DEFINE_string("config_file", os.path.join("configs", "rnn_config_file"), "模型配置文件")
flags.DEFINE_string("train_dir", os.path.join("data", "yinsi_textcnn_train.txt"), "训练集路径")
flags.DEFINE_string("val_dir", os.path.join("data", "yinsi_textcnn_test.txt"), "验证集路径")
flags.DEFINE_string("test_dir", os.path.join("data", "yinsi_textcnn_test.txt"), "测试集路径")
flags.DEFINE_string("vocab_dir", os.path.join("data", "yinsi_rnn_vocab.txt"), "词汇表路径")
flags.DEFINE_string("save_dir", os.path.join("checkpoints/textcnn", "best_validation"), "最佳验证结果保存路径")

FLAGS = tf.app.flags.FLAGS
assert 0 <= FLAGS.dropout_keep_prob < 1, "dropout rate between 0 and 1"
assert FLAGS.learning_rate > 0, "learning rate must larger than 0"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]
assert FLAGS.rnn in ["gru", "LSTM"]


def main_train():
    # 如果不存在词汇表则新建
    if not os.path.exists(FLAGS.vocab_dir):
        build_vocab(FLAGS.train_dir, FLAGS.vocab_dir, FLAGS.vocab_size)
    id_to_word, word_to_id = read_vocab(FLAGS.vocab_dir)

    if not os.path.exists(FLAGS.category_dir):
        build_category(FLAGS.train_dir, FLAGS.category_dir)
    id_to_cat, cat_to_id = read_category(FLAGS.category_dir)

    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = create_rnn_config_model(FLAGS, id_to_word)

    model = TextRNN(config)

    train(model, config, word_to_id, cat_to_id)


def main_test():
    id_to_word, word_to_id = read_vocab(FLAGS.vocab_dir)
    id_to_cat, cat_to_id = read_category(FLAGS.category_dir)
    config = load_config(FLAGS.config_file)
    model = TextRNN(config)

    test(model, config, word_to_id, cat_to_id, id_to_cat)


def save_rnn_for_java():
    config = load_config(FLAGS.config_file)
    model = TextRNN(config)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=FLAGS.save_path)  # 读取保存的模型

        builder = tf.saved_model.builder.SavedModelBuilder("tmp/rnn_model")
        builder.add_meta_graph_and_variables(
            session,
            [tf.saved_model.tag_constants.SERVING]
        )
        builder.save()


if __name__ == "__main__":
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
            mkdir()
        main_train()
    else:
        main_test()
        # evaluate_one()

    # save_rnn_for_java()
