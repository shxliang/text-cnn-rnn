import os
import shutil

import time
from datetime import timedelta


def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """

    if os.path.isdir(params.output_dir):
        shutil.rmtree(params.output_dir)

    if os.path.isdir(params.tensorboard_dir):
        shutil.rmtree(params.tensorboard_dir)

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")


def get_time_dif(start_time):
    """
    获取已使用时间
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def mkdir(params):
    if not os.path.isdir(params.output_dir):
        os.makedirs(params.output_dir)

    if not os.path.isdir(params.tensorboard_dir):
        os.makedirs(params.tensorboard_dir)
