import tensorflow as tf
import tensorflow.contrib.keras as kr

from data_loaders.data_loader import read_vocab, read_category
from utils.config_utils import load_config


def evaluate_one(Model, config_file):
    """
    加载模型进行单条样本预测
    :param Model:
    :param config_file:
    :return:
    """
    config = load_config(config_file)
    id_to_word, word_to_id = read_vocab(config.vocab_dir)
    id_to_cat, cat_to_id = read_category(config.category_dir)
    model = Model(config)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path="checkpoints/textcnn/best_validation")  # 读取保存的模型

        while True:
            try:
                line = input("请输入测试文本: ")
                if line == "exit":
                    exit(0)

                data = [word_to_id[x] if x in word_to_id else word_to_id["<UNK>"] for x in line]
                pad_data = kr.preprocessing.sequence.pad_sequences([data], config.seq_length, padding="post",
                                                                   truncating="post")
                print(pad_data)
                feed_dict = {
                    model.input_x: pad_data,
                    model.keep_prob: 1.0
                }
                y_pred_cls, logits = session.run([model.y_pred_cls, model.logits], feed_dict=feed_dict)
                print(y_pred_cls[0], tf.nn.softmax(logits).eval())
                print("所属类别: {}".format(id_to_cat[y_pred_cls[0]]))
            except Exception as e:
                print(e)
