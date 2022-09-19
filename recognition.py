import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np
import random

SEED = 1

random.seed(SEED)
np.random.seed(SEED)


def listdir(path):
    """os.listdir with ".DS_Store" removed

    :param path: a str, bytes, or a path-like object
    :return: a list containing the names of the files in the directory.
    """
    ls = os.listdir(path)
    try:
        ls.remove(".DS_Store")
    except ValueError as e:
        pass
    return ls


def load_song(song_folder):
    song_specs = []  # array shape (1000, 1280, 128)
    idx_to_genre = []  # {g for g in ls}
    genre_to_idx = {}  # {v: k for k, v in enumerate(ls)}
    genres = []  # [j for j in range(10) for i in range(10)]

    for genre in listdir(song_folder):
        genre_to_idx[genre] = len(genre_to_idx)
        idx_to_genre.append(genre)
        genre_folder = os.path.join(song_folder, genre)
        for song in listdir(genre_folder):
            if song.endswith(".wav"):
                signal, sr = librosa.load(
                    os.path.join(genre_folder, song))
                melspec = librosa.feature.melspectrogram(
                    signal, sr=sr).T[:1100, ]
                song_specs.append(melspec)
                genres.append(genre_to_idx[genre])

    return song_specs, genres, genre_to_idx, idx_to_genre


song_specs, genres, genre_to_idx, idx_to_genre = load_song("./data/music")

plt.rcParams["figure.figsize"] = (12, 8)


def show_spectrogram(genre_name):
    genre_idx = genre_to_idx[genre_name]
    specs = []
    for spec, idx in zip(song_specs, genres):
        if idx == genre_idx:
            specs.append(spec)
            if len(specs) >= 25:
                break
    if not specs:
        raise ValueError("specs not found")
    x = np.concatenate(specs, axis=1)
    x = (x - x.min()) / (x.max() - x.min())
    x = (x * 20).clip(0, 1.0)
    plt.imshow(x)
    plt.colorbar()
    plt.show()


show_spectrogram("pop")
tf.random.set_seed(SEED)


def cnn_model(input_shape):
    inputs = Input(input_shape)
    x = inputs

    # 一维卷积池化堆叠
    levels = 64
    for level in range(3):
        x = Conv1D(levels, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, strides=2)(x)
        levels *= 2

    # Global -> shape(128)
    x = GlobalMaxPooling1D()(x)

    # 计算类型标签的全连接网络
    for fc in range(2):
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

    labels = Dense(10, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[labels])

    # optimizer and compile model
    sgd = SGD(learning_rate=0.0003, momentum=0.9, decay=1e-5, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = cnn_model((128, 128))
model.summary()
