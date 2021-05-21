import logging
import torch
import math
import numpy as np
import json
import os

def helper_function():
    print("helper_function")


def sizeof_fmt(num, spacing=True, suffix='B'):
    # turns bytes object into human readable
    if spacing:
        fmt = "{:>7.2f}{:<3}"
    else:
        fmt = "{:.2f}{}"

    for unit in ['', 'Ki', 'Mi']:
        if abs(num) < 1024.0:
            return fmt.format(num, unit + suffix)
        num /= 1024.0
    return fmt.format(num, 'Gi' + suffix)

def cache_stats(human_readable=True, spacing=True):
    if not torch.cuda.is_available():
        return 0
    # returns current allocated torch memory
    if human_readable:
        return sizeof_fmt(torch.cuda.memory_reserved(), spacing)
    else:
        return int(torch.cuda.memory_reserved())

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
