import logging
import math
import numpy as np
import json
import os


# === LOGGING =============================================================================================
def setup_logger(logger_name, filename, mode, terminator):
    l = logging.getLogger(filename)
    formatter = logging.Formatter()
    file_handler = logging.FileHandler(filename, mode=mode)
    file_handler.setFormatter(formatter)
    # file_handler.terminator = terminator

    l.setLevel(logging.INFO)
    l.addHandler(file_handler)
    return l


def log_print_curry(loggers):
    def log_print(string, end='\n', flush=False):
        print(string, end=end, flush=flush)
        for logger in loggers:
            logger.info(string)
    return log_print


# === SOME I/O HELPERS =================================================================================
def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)


# === DATA LOADING HELPERS ====================================================
def get_dataset_paths(data_dir):
    paths = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if 'dataset' in d])
    return paths


def load_dataset_metadata(dataset_path):
    with open(os.path.join(dataset_path, 'dataset_metadata'), "r") as f:
        metadata = json.load(f)
    return metadata


# load dataset from location data/$dataset/
def load_datasets(data_path):
    train_x = np.load(os.path.join(data_path,'train_x.npy'))
    train_y = np.load(os.path.join(data_path,'train_y.npy'))
    valid_x = np.load(os.path.join(data_path,'valid_x.npy'))
    valid_y = np.load(os.path.join(data_path,'valid_y.npy'))
    test_x = np.load(os.path.join(data_path,'test_x.npy'))
    metadata = load_dataset_metadata(data_path)

    return (train_x, train_y), \
           (valid_x, valid_y), \
           (test_x), metadata