# coding: utf-8

import tensorflow as tf


class TextCNN(object):
    """
    文本分类，CNN模型
    """

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """
        CNN模型
        """
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # 1维卷积层
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            # [batch_size, (seq_length - kernel_size) + 1, num_filters]
            gmp = tf.reduce_max(conv, axis=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 全连接层，作为分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.softmax = tf.nn.softmax(self.logits)

            # 预测类别
            self.y_pred_cls = tf.argmax(self.softmax, 1)

            # 二分类，当正例比负例概率大于一定阈值时才判为正
            # def true_f():
            #     return 1
            #
            # def false_f():
            #     return 0
            #
            # self.y_pred_cls = tf.cond(
            #     tf.greater(tf.subtract(self.softmax[-1][1], self.softmax[-1][0]), tf.constant(0.4)), true_f, false_f)

        with tf.name_scope("optimize"):
            # 损失函数: 交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器: Adam
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 正确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), tf.cast(self.y_pred_cls, tf.int64))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
