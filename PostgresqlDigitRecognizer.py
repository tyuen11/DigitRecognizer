import psycopg2
import csv
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import *
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def  modelCreateAndPredict(image_to_predict):
    image_to_predict = np.array(image_to_predict).reshape(28, 28)
    image_to_predict = image_to_predict.reshape([-1, 28, 28, 1])  # make image_array into a 4D data set

    connection = psycopg2.connect(user="postgres",
                                  password="password",
                                  host="127.0.0.1",
                                  database="myDatabase")

    cursor = connection.cursor()

    train_data = pd.read_sql_query('select * from train', connection)
    train_data = train_data.drop(columns=['label'])
    train_data = train_data / 255
    train_data = train_data.values.reshape(42000, 28, 28)
    train_data = train_data.reshape([-1, 28, 28, 1])  # make image_array into a 4D data set


    # get list of all the labels so we can one hot encode it
    cursor.execute('select label from train')
    labels = cursor.fetchall()
    train_labels = []
    for label in labels:
        train_labels.append(label[0])
    train_labels = np.array(train_labels)

    one_hot_labels = np.zeros((train_labels.size, train_labels.size + 1))
    one_hot_labels[np.arange(train_labels.size), train_labels] = 1

    model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            Flatten(),
            Dense(10, activation='softmax'),
        ])
    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    train_labels_bin_matr = to_categorical(train_labels)

    model.fit(train_data, train_labels_bin_matr, validation_split=0.30, epochs=1,
              batch_size=10, shuffle=True, verbose=2)

    # save the model
    model.save('model.h5')

    predictions = model.predict(image_to_predict, verbose=0)
    # print(predictions)
    answ = []
    for x in predictions:
        answ.append(np.argmax(x))
        print(np.argmax(x))  # get the index of the max number in the array

    cursor.close()
    connection.close()
    return answ[0]


if __name__ == "__main__":
    test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 169, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 188, 254, 133, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 159, 253, 104, 2, 0, 0, 0, 57, 74, 108, 155, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78,
            253, 254, 244, 242, 242, 242, 242, 252, 254, 254, 199, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 224,
            254, 254, 254, 254, 254, 247, 189, 153, 104, 70, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 252,
            254, 128, 64, 21, 21, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 234, 242, 75, 4, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 234, 151, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 234, 197, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 204, 254, 221, 126, 44, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 11, 139, 237, 254, 254, 206, 85, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 26, 64, 207, 249, 254, 167, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            56, 223, 254, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 254, 142,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 254, 142, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 47, 2, 0, 0, 0, 0, 0, 3, 189, 254, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 137, 219, 11, 0, 0, 0, 0, 56, 182, 254, 194, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            222, 13, 0, 8, 9, 56, 141, 244, 253, 163, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 247, 181,
            177, 248, 254, 255, 252, 206, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 227, 249,
            234, 168, 168, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
    modelCreateAndPredict(test)
