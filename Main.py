import os

import numpy as np
import pydub
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Dropout, BatchNormalization


def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2 ** 15
    else:
        return a.frame_rate, y


def to_categorical(vector, nb_classes):
    if not nb_classes:
        nb_classes = np.max(vector) + 1
    Y = np.zeros((len(vector), nb_classes))
    for id, item in enumerate(vector):
        Y[id][0] = vector[id]
        Y[id][1] = 1 - vector[id]
    return Y


def addVectorsToTest(x, y, vector, res):
    addVector(res, x, np.array(vector[:, 0], dtype=np.float32).reshape(660096, ), y)
    addVector(res, x, np.array(vector[:, 1], dtype=np.float32).reshape(660096, ), y)


def addVector(anwser, x_result, vector, y_result):
    for i in range(0, last_index, count_shags):
        x_result.append(vector[i:i + shag])
        y_result.append(anwser)


def readFolder(folder_name, x, y, answer):
    for root, dirs, files in os.walk(folder_name):
        for filename in files:
            sr, vector = read(folder_name + '/' + filename)
            vector = (vector + 32768) / 65535
            if np.array(vector).shape[0] == 660096:
                addVectorsToTest(x, y, vector, answer)


full_len = 660096
len_kyrs = 2

shag = (int)(full_len / 5)
last_index = (int)(full_len - shag)
count_shags = (int)(full_len / 150)
n = shag
m = len_kyrs
r = pow(n / m, 1 / 3)
k1 = (int)(m * pow(r, 2))
k2 = (int)(m * r)

x_train = []
y_train = []

x_test = []
y_test = []

readFolder("./0", x_train, y_train, 0)
readFolder("./1", x_train, y_train, 1)
readFolder("./05", x_train, y_train, 0.5)

readFolder("./0t", x_test, y_test, 0)
readFolder("./1t", x_test, y_test, 1)

y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)
x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)


def build_model(learning_rate=0.00001):
    model = keras.Sequential()
    model.add(Dense(1000, activation='relu', input_shape=(shag,), kernel_regularizer=keras.regularizers.l1(0.01),
    activity_regularizer=keras.regularizers.l2(0.3)))
    # model.add(Dense(k1/2, activation='relu', input_shape=(shag,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='relu', input_shape=(shag,)))
    # model.add(Dense(k1, activation='relu', activity_regularizer=keras.regularizers.l1_l2(l1=0.1, l2=0.01)))
    # model.add(Dense(k1, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(k2, activation='relu',kernel_regularizer=keras.regularizers.l1(0.01),
    activity_regularizer=keras.regularizers.l2(0.3)))
    # model.add(Dense(k2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(len_kyrs, activation='softmax'))

    # model.add(keras.layers.Activation('softmax'))
    # tf.reset_default_graph()
    # net1 = tflearn.input_data([None, shag])
    # net1 = tflearn.batch_normalization(net1)
    # net1 = tflearn.fully_connected(net1, k1, regularizer='L2')
    # net1 = tflearn.dropout(net1, 0.8)
    # net1 = tflearn.fully_connected(net1, k2, regularizer='L2')
    # net1 = tflearn.dropout(net1, 0.8)
    # net1 = tflearn.fully_connected(net1, len_kyrs, activation='softmax')
    # net1 = tflearn.regression(
    #     net1,
    #     optimizer='adam',
    #     learning_rate=learning_rate,
    #     loss='binary_crossentropy')

    # model = tflearn.DNN(net1)
    return model


model = build_model()

loss_fn = keras.losses.SparseCategoricalCrossentropy()
opt = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=500, epochs=1)

result = model.predict(x_test)

_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))
#
# shag = 100 / result.shape[0]
# percentage = 0.0
# for idx, val in enumerate(result):
#     if val[0] > 0.5 and y_test[idx][0] > 0.5 or val[0] < 0.5 and y_test[idx][0] < 0.5:
#         percentage += shag
# print("percentage = " + str(percentage))
# date_after_month = datetime.now()
# name_model = "my_model." + date_after_month.strftime('%Y.%m.%d_%H.%M.%S')
# model.save('checkpoints/' + name_model + '/' +str(int(percentage)) + "_"+ name_model + '.tflearn')
t = 3
