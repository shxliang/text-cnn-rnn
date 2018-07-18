import os
import shutil

import time
from datetime import timedelta


def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(params.vocab_dir):
        os.remove(params.vocab_dir)

    if os.path.isdir(params.save_dir):
        shutil.rmtree(params.save_dir)

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")


def get_time_dif(start_time):
    """
    获取已使用时间
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def mkdir():
    if not os.path.isdir("configs"):
        os.mkdir("configs")
