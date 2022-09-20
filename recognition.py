import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from sklearn.metrics import confusion_matrix

SEED = 1

random.seed(SEED)
np.random.seed(SEED)
SIZE = 1050


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
    song_specs = []
    idx_to_genre = []
    genre_to_idx = {}
    genres = []

    for genre in listdir(song_folder):
        genre_to_idx[genre] = len(genre_to_idx)
        idx_to_genre.append(genre)
        genre_folder = os.path.join(song_folder, genre)
        for song in listdir(genre_folder):
            if song.endswith(".wav"):
                signal, sr = librosa.load(
                    os.path.join(genre_folder, song))
                melspec = librosa.feature.melspectrogram(
                    signal, sr=sr).T[:SIZE, ]
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
            if len(specs) >= 10:
                break
    if not specs:
        raise ValueError("specs not found")
    x = np.concatenate(specs, axis=1)
    x = (x - x.min()) / (x.max() - x.min())
    x = (x * 20).clip(0, 1.0)
    plt.imshow(x)
    plt.colorbar()
    plt.show()


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

    # Global -> shape(105)
    x = GlobalMaxPooling1D()(x)

    # 计算类型标签的全连接网络
    for fc in range(2):
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)

    labels = Dense(len(idx_to_genre), activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[labels])

    # optimizer and compile model
    sgd = SGD(learning_rate=0.0003, momentum=0.9, decay=1e-5, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = cnn_model((SIZE // 10, 128))
model.summary()


def split_10(x, y):
    s = x.shape
    s = (s[0] * 10, s[1] // 10, s[2])
    return x.reshape(s), np.repeat(y, 10, axis=0)


test_size = 0.2
genres_one_hot = to_categorical(genres, num_classes=len(genre_to_idx))
x_train, x_test, y_train, y_test = train_test_split(
    np.array(song_specs), np.array(genres_one_hot),
    test_size=test_size, stratify=genres)
x_train, y_train = split_10(x_train, y_train)
x_test, y_test = split_10(x_test, y_test)

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=3,
                           verbose=0,
                           mode='auto')

history = model.fit(x_train, y_train,
                    batch_size=SIZE // 10,
                    epochs=int(len(idx_to_genre)*10*test_size),
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop])
model.save("music_recognition.h2")


def unsplit(values):
    chunks = np.split(values, 10)
    return np.array([np.argmax(chunk) % len(idx_to_genre) for chunk in chunks])


pred_values = model.predict(x_test)
predictions = unsplit(pred_values)
truth = unsplit(y_test)
print("accuracy_score:", accuracy_score(predictions, truth))
cm = confusion_matrix(np.argmax(pred_values, axis=1), np.argmax(y_test, axis=1))
print(cm)
plt.imshow(cm.T, interpolation='nearest', cmap='gray')
plt.xticks(np.arange(0, len(idx_to_genre)), idx_to_genre)
plt.yticks(np.arange(0, len(idx_to_genre)), idx_to_genre)

plt.show()
