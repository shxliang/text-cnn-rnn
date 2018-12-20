# coding: utf-8

from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
from sklearn import metrics

from data_loaders.data_loader import batch_iter, process_file
from utils.utils import get_time_dif


def create_feed_dict(model, x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, model, x_, y_):
    """
    评估在某一数据上的正确率和损失
    """
    data_size = len(x_)
    batch_eval = batch_iter(x_, y_, 128)

    total_loss = 0.0
    total_acc = 0.0

    for x_batch, y_batch in batch_eval:
        batch_size = len(x_batch)
        # 进行预测时不使用dropout
        feed_dict = create_feed_dict(model, x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_size
        total_acc += acc * batch_size

    return total_loss / data_size, total_acc / data_size


def train(model, config, word_to_id, cat_to_id):
    print("Configuring TensorBoard and Saver...")
    # 配置Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textrnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    # 设置TensorBoard中需要统计的指标
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置Saver，用于保存模型
    saver = tf.train.Saver()
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    print("Loading training and validation data_loaders...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(config.train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(config.val_dir, word_to_id, cat_to_id, config.seq_length)
    # 计算数据载入时间
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # 将计算图添加到TensorBoard中用于展示
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    # 已运行的总批次
    total_batch = 0
    # 最佳验证集准确率
    best_acc_val = 0.0
    # 记录上一次提升时是第几个批次
    last_improved = 0
    # 如果超过1000个批次未提升，则提前结束训练
    require_improvement = 1000

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = create_feed_dict(model, x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, model, x_val, y_val)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    # 存储模型
                    saver.save(sess=session, save_path=config.save_dir)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test(model, config, word_to_id, cat_to_id, id_to_cat):
    print("Loading test data_loaders...")
    start_time = time.time()
    x_test, y_test = process_file(config.test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # 读取保存的模型
    saver.restore(sess=session, save_path=config.save_dir)

    print('Testing...')
    loss_test, acc_test = evaluate(session, model, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    # 用于保存预测结果
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)
    # 逐批次处理
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # PRF评估
    print("PRF:")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=id_to_cat))

    # 混淆矩阵
    print("Confusion Matrix:")
    print("Truth \ Prediction")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
