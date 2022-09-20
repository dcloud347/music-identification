import os
import random

import librosa
import numpy as np
import tensorflow as tf
from utils import listdir, unsplit

SEED = 1
SIZE = 1200
idx_to_genre = ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hiphop', 'country', 'jazz']

model = tf.keras.models.load_model('music_recognition.h2')

random.seed(SEED)
np.random.seed(SEED)


def test(genre):
    path = "./data/music/" + genre
    signal, sr = librosa.load(
        os.path.join(path, listdir(path)[random.randint(0, 100)]))
    test_data = librosa.feature.melspectrogram(
        signal, sr=sr).T[:SIZE, ]
    return test_data


def split_10(x):
    s = x.shape
    s = (s[0] * 10, s[1] // 10, s[2])
    return x.reshape(s)


def find_mode(lst):
    return max(lst, default='列表为空', key=lambda v: lst.count(v))


test_data = np.array([test("reggae")])
test_data = split_10(test_data)
predict_result = model.predict(test_data)
predictions = unsplit(predict_result)
print(idx_to_genre[find_mode(list(predictions))])
