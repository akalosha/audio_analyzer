import csv
import pathlib
import warnings

import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8, 8))
genres = '1 0'.split()


def made_vectors(audio_folder_name, image_folder_name, csv_folder_name, only_read):
    if not only_read:
        for g in genres:
            pathlib.Path(f'{image_folder_name}/{g}').mkdir(parents=True, exist_ok=True)
            for filename in os.listdir(f'./{audio_folder_name}/{g}'):
                songname = f'./{audio_folder_name}/{g}/{filename}'
                y, sr = librosa.load(songname, mono=True, duration=5)
                plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default',
                             scale='dB')
                plt.axis('off')
                plt.savefig(f'{image_folder_name}/{g}/{filename[:-3].replace(".", "")}.png')
                plt.clf()
        header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header += ' label'
        header = header.split()
        file = open(csv_folder_name, 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        for g in genres:
            for filename in os.listdir(f'./{audio_folder_name}/{g}'):
                songname = f'./{audio_folder_name}/{g}/{filename}'
                y, sr = librosa.load(songname, mono=True, duration=30)
                rmse = librosa.feature.rmse(y=y)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                to_append += f' {g}'
                file = open(csv_folder_name, 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())

    data = pd.read_csv(csv_folder_name)
    data.head()
    # Удаление ненужных столбцов
    data = data.drop(['filename'], axis=1)
    # Создание меток
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    # Масштабирование столбцов признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
    return X, y


only_read = False

X_train, y_train = made_vectors('mp3', 'img_data', 'dataset.csv', only_read)
X_test, y_test = made_vectors('mp3_test', 'img_data_test', 'dataset_test.csv', only_read)

model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
import os

classifier = model.fit(X_train, y_train, epochs=20)

result = model.predict(X_test)

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))
y = 2
