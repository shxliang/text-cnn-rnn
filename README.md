在原作者(https://github.com/gaussic/text-classification-cnn-rnn.git) 的基础上进行了修改和优化:
- 基于DL Project Template重构代码
- vocab增加"\<UNK>"，对vocab外的字用"\<UNK>"代替
- padding和truncating使用post方式
- 修改模型参数设置方式，输出配置文件


# Text Classification with CNN and RNN

使用卷积神经网络以及循环神经网络进行中文文本分类

CNN做句子分类的论文可以参看: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

还可以去读dennybritz大牛的博客：[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

以及字符级CNN的论文：[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

本文是基于TensorFlow在中文数据集上的简化实现，使用了字符级CNN和RNN对中文文本进行分类，达到了较好的效果。

文中所使用的Conv1D与论文中有些不同，详细参考官方文档：[tf.nn.conv1d](https://www.tensorflow.org/api_docs/python/tf/nn/conv1d)

## 环境

- Python3
- TensorFlow1.3 以上
- numpy
- scikit-learn
- scipy

## 数据集

使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。

本次训练使用了其中的10个分类，每个分类6500条数据。

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```

这个子集可以在此下载：链接:[http://pan.baidu.com/s/1bpq9Eub](http://pan.baidu.com/s/1bpq9Eub)  密码:ycyw

数据集划分如下：

- 训练集: 5000 * 10
- 验证集: 500 * 10
- 测试集: 1000 * 10

从原数据集生成子集的过程请参看`utils`下的两个脚本。其中，`copy_data.sh`用于从每个分类拷贝6500个文件，`cnews_group.py`用于将多个文件整合到一个文件中。执行该文件后，得到三个数据文件：

- cnews.train.txt: 训练集(50000条)
- cnews.val.txt: 验证集(5000条)
- cnews.test.txt: 测试集(10000条)

## 预处理

`data_loaders/cnews_loader.py`为数据的预处理文件。

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `preocess_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

经过数据预处理，数据的格式如下：

| Data | Shape | Data | Shape |
| :---------- | :---------- | :---------- | :---------- |
| x_train | [50000, 600] | y_train | [50000, 10] |
| x_val | [5000, 600] | y_val | [5000, 10] |
| x_test | [10000, 600] | y_test | [10000, 10] |

## CNN卷积神经网络

### 配置项

CNN可配置的参数在`main_cnn.py`中，配置文件输出在`configs`中。

### CNN模型

具体参看`models/cnn_model.py`的实现。

大致结构如下：

![images/cnn_architecture](doc/cnn_architecture.png)

### 训练与测试

在`main_cnn.py`中可以进行模型的训练和测试。

训练时设置为
```
flags.DEFINE_boolean("clean", True, "Wither clean train folder")
flags.DEFINE_boolean("train", True, "Wither train the model")
```

测试时设置为，可以进行单样本交互测试
```
flags.DEFINE_boolean("clean", False, "Wither clean train folder")
flags.DEFINE_boolean("train", False, "Wither train the model")
```

## RNN循环神经网络

### 配置项

RNN可配置的参数如下所示，在`main_rnn.py`中，配置文件输出在`configs`中。

### RNN模型

具体参看`models/rnn_model.py`的实现。

大致结构如下：

![images/rnn_architecture](doc/rnn_architecture.png)

### 训练与测试

> 这部分的代码与 run_cnn.py极为相似，只需要将模型和部分目录稍微修改。

在`main_rnn.py`中可以进行模型的训练和测试。

训练时设置为
```
flags.DEFINE_boolean("clean", True, "Wither clean train folder")
flags.DEFINE_boolean("train", True, "Wither train the model")
```

测试时设置为，可以进行单样本交互测试
```
flags.DEFINE_boolean("clean", False, "Wither clean train folder")
flags.DEFINE_boolean("train", False, "Wither train the model")
```