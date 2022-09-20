import os

import numpy as np


def unsplit(values):
    chunks = np.split(values, 10)
    return np.array([np.argmax(chunk) % 10 for chunk in chunks])


def listdir(path):
    ls = os.listdir(path)
    try:
        ls.remove(".DS_Store")
    except ValueError as e:
        pass
    return ls
