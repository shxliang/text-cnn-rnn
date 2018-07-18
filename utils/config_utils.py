import json
import codecs
from bunch import Bunch
from collections import OrderedDict


def create_cnn_config_model(params, id_to_word):
    config_dict = OrderedDict()

    config_dict["embedding_dim"] = params.embedding_dim
    config_dict["seq_length"] = params.seq_length
    config_dict["num_classes"] = params.num_classes
    config_dict["num_filters"] = params.num_filters
    config_dict["kernel_size"] = params.kernel_size
    config_dict["vocab_size"] = len(id_to_word)
    config_dict["hidden_dim"] = params.hidden_dim

    config_dict["learning_rate"] = params.learning_rate
    config_dict["dropout_keep_prob"] = params.dropout_keep_prob
    config_dict["batch_size"] = params.batch_size
    config_dict["num_epochs"] = params.num_epochs
    config_dict["print_per_batch"] = params.print_per_batch
    config_dict["save_per_batch"] = params.save_per_batch
    config_dict["optimizer"] = params.optimizer

    config_dict["tensorboard_dir"] = params.tensorboard_dir
    config_dict["train_dir"] = params.train_dir
    config_dict["val_dir"] = params.val_dir
    config_dict["test_dir"] = params.test_dir
    config_dict["save_dir"] = params.save_dir

    save_config(config_dict, params.config_file)
    config = Bunch(config_dict)
    return config


def create_rnn_config_model(params, id_to_word):
    config_dict = OrderedDict()

    config_dict["embedding_dim"] = params.embedding_dim
    config_dict["seq_length"] = params.seq_length
    config_dict["num_classes"] = params.num_classes
    config_dict["vocab_size"] = len(id_to_word)
    config_dict["num_layers"] = params.num_layers
    config_dict["hidden_dim"] = params.hidden_dim
    config_dict["rnn"] = params.rnn

    config_dict["learning_rate"] = params.learning_rate
    config_dict["dropout_keep_prob"] = params.dropout_keep_prob
    config_dict["batch_size"] = params.batch_size
    config_dict["num_epochs"] = params.num_epochs
    config_dict["print_per_batch"] = params.print_per_batch
    config_dict["save_per_batch"] = params.save_per_batch
    config_dict["optimizer"] = params.optimizer

    config_dict["tensorboard_dir"] = params.tensorboard_dir
    config_dict["train_dir"] = params.train_dir
    config_dict["val_dir"] = params.val_dir
    config_dict["test_dir"] = params.test_dir
    config_dict["save_dir"] = params.save_dir

    save_config(config_dict, params.config_file)
    config = Bunch(config_dict)
    return config


def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with codecs.open(config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = Bunch(config_dict)
    return config


def save_config(config_dict, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with codecs.open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
