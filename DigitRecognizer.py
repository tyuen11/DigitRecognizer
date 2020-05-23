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

# Todo: Need to add the train.csv file into Django so others can execute modelCreateAnePredict()

def  modelCreateAndPredict(image_to_predict):

    # See what the number we are trying to predict is
    image_to_predict_picture = np.array(image_to_predict).reshape(28, 28)
    plt.imshow(image_to_predict_picture, cmap='Greys')
    plt.show()

    image_to_predict = np.array(image_to_predict).reshape(28, 28)
    # plt.imshow(image_to_predict, cmap='Greys')
    # plt.show()
    image_to_predict = image_to_predict.reshape([-1,28,28,1]) # make image_array into a 4D data set


    # Data pre-processing
    #  print(train_data.isnull().any().describe())  # No missing values
    #  print(train_data.isna().any().describe())  # No NaN values

    # Create array of arrays of each number images' pixel values
    with open("digit-recognizer/train.csv", "r") as td:
        image_array = []  # The n-D array that will contain the pixel values
        train_label = []
        # Create a reader object that will iterate over lines in test_data
        reader = csv.reader(td, delimiter="\t")
        # Loop through the row and append each row in image_array
        for i, line in enumerate(reader):
            # line is a one element array of the while line in the csv file for each line
            # Skip the first row as it is just the column names
            if (line[0])[0:1].isalpha():
                continue
            # Remove '' and make the result into an int array and make the line into an int array
            # Remove the first number as it is the number the pixels create
            pixel_string = (line[0])[2:]
            # Enter the first number into our train_label (MIGHT NOT BE NEEDED)
            train_label.append(ord(line[0][0])-48)
            pixel_array = [int(x)/256 for x in pixel_string.split(",")] # make each value between 0 and 1 since current values represent the RGB pixel values (0 to 1 corresponds to colors between black and white
            # After getting the array of pixel arrays, we need to make an actual 28 x 28 array for each pixel array
            pixel_array = np.array(pixel_array).reshape(28,28)
            image_array.append(pixel_array)

    # Convert the train_label array to a numpy array so we can one-hot encode it
    train_label = np.array(train_label)
    #print(train_label)
    one_hot_labels = np.zeros((train_label.size, train_label.size+1))
    one_hot_labels[np.arange(train_label.size), train_label] = 1

    # Create the CNN model
    # relu is a rectified liner unit layer
    '''
    cnn_model = Sequential([
        Dense(16, activation='relu', input_shape=(28, 28, 1)), 
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
        Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        Flatten(),
        Dense(2, activation='softmax'),
    
    ])'''

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Flatten(),
        Dense(10 , activation='softmax'),
    ])
    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Use catergorical_crossentropy as we did a one hot encoding of the train labels
    # https://www.reddit.com/r/MLQuestions/comments/93ovkw/what_is_sparse_categorical_crossentropy/
    # ^^ link for sparse_categorical_crossentropy and categorical_crossentropy

    image_array = np.array(image_array)
    image_array = image_array.reshape([-1, 28, 28, 1])  # make image_array into a 4D data set
    train_label_bin_matr = to_categorical(train_label)

    model.fit(image_array, train_label_bin_matr, validation_split=0.30, epochs=1,
                     batch_size=10, shuffle=True, verbose=2)
    # Since our training dataset is large, we will train the model through a series of batches of data. In our case our
    # batch size is 10 datapoints per run.

    # Preprocess the test data
    test_data = pd.read_csv('/Users/timyuen/PycharmProjects/DigitRecognizer/digit-recognizer/test.csv')  # Another way to get the data from csv
    test_data = test_data / 255 # Make the value of each pixel color between 0 and 1 (greyscale pixels)
    print(test_data)
    test_data = test_data.values.reshape(28000, 28, 28)
    test_data = test_data.reshape([-1, 28, 28, 1])  # make image_array into a 4D data set
    print(test_data)

    predictions = model.predict(image_to_predict, verbose=0)
    # print(predictions)
    answ = []
    for x in predictions:
        answ.append(np.argmax(x))
        print(np.argmax(x))  # get the index of the max number in the array

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


# -----------------------------------------------------------------------------------------------
# Function to preprocess data set with imputation using mean (WILL NOT BE USED)
# Imputation looks for all empty entries in the data set and inserts the column's mean into the empty entry
def mean_imputation(data):
    imputer = SimpleImputer()  # create instance of Simple Imputer
    imputed_X = pd.DataFrame(imputer.fit_transform(data))  # fit and transform the data using the mean
    # Using the Simple Imputer removes the column names, need to put them back into the data set
    imputed_X.columns = data.columns
    return imputed_X  # return the imputed data set


# Function to preprocess data set with imputation using mean (WILL NOT BE USED)
# Imputation looks for all empty entries in the data set and inserts the column's median into the empty entry
def median_imputation(data):
    imputer = SimpleImputer(strategy='median')  # create instance of Simple Imputer
    imputed_X = pd.DataFrame(imputer.fit_transform(data))  # fit and transform the data using the mean
    # Using the Simple Imputer removes the column names, need to put them back into the data set
    imputed_X.columns = data.columns
    return imputed_X  # return the imputed data set

# One determines whether to use mean_imputation or median_imputation by finding its MAE and choosing the one the has the
# lowest MAE
# The MAE is determined by creating a model with the imputated data set and checking the error between the actual and
# predicted values
# -----------------------------------------------------------------------------------------------

