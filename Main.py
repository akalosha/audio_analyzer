
print('Start project')
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import pydub
def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y


x = []
yy = []

for root, dirs, files in os.walk("./0"):
    for filename in files:
        sr, xx = read("./0/" + filename)
        if np.array(xx).shape[0] == 660096:
            x.append(np.array(xx[:, 0], dtype=np.float32).reshape(660096, ))
            # x.append(xx[:,0])
            yy.append(0)

for root, dirs, files in os.walk("./1"):
    for filename in files:
        sr, xx = read("./1/" + filename)
        if np.array(xx).shape[0] == 660096:
            x.append(np.array(xx[:, 0], dtype=np.float32).reshape(660096, ))
            yy.append(1)
yyy = np.array(yy, dtype=np.float32)
y = np.reshape(yyy, (len(yyy), 1))
x = np.array(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# inputs = keras.Input(shape=(784,), name='digits')
# x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
# x = layers.Dense(64, activation='relu', name='dense_2')(x)
# outputs = layers.Dense(10, name='predictions')(x)
full_len=660096
len_kyrs = 2
n = full_len
m = len_kyrs
r = pow(n / m, 1 / 3)
k1 = (int)(m * pow(r, 2))
k2 = (int)(m * r)

import tensorflow as tf;
from tensorflow import keras;


# def build_model(learning_rate=0.1):
#     tf.reset_default_graph()
#     x = tflearn.input_data([None, full_len])
#     # net = tflearn.batch_normalization(x)
#     # net1 = tflearn.fully_connected(x, 1, activation='LeakyReLU')
#     # net2 = tflearn.batch_normalization(net1)
#     # net3 = tflearn.fully_connected(net2, k2, activation='LeakyReLU')
#     net4 = tflearn.fully_connected(x, len_kyrs, activation='softmax')
#     regression = tflearn.regression(
#         net4,
#         optimizer='sgd',
#         learning_rate=learning_rate,
#         loss='categorical_crossentropy')
#     model = tflearn.DNN(net4, tensorboard_dir='/tmp/tflearn_logs8/',
#                         tensorboard_verbose=0)
#     return model
# model = build_model()
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(sparse=False) # Key here is sparse=False!
# y_categorical = enc.fit_transform(y.reshape((y.shape[0]),1))
#
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(x, y_categorical, nb_epoch=20, batch_size=10)\


model = keras.Sequential()
model.add(keras.layers.Dense(1, activation='relu', input_shape=(full_len,)))
# model.add(keras.layers.Dense(8, activation='relu'))
# model.add(keras.layers.Dense(8, activation='sigmoid'))
model.add(keras.layers.Activation('softmax'))
# adam = keras.optimizers.Adam()
# model.compile(optimizer=adam, loss='categorical_crossentropy')
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[keras.metrics.categorical_accuracy])

model.fit(X_train, y_train, epochs=5, batch_size=2)
a = model.predict(X_test)
t =3