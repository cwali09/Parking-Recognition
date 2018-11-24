import random
import os.path
from os import listdir
from sys import platform

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import keras.models
from keras.layers import Dropout

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pandas as pd

from data_retrieval import DataRetrieval
from preprocessing import Preprocessing
from CONSTANTS import IMG_CHANNEL, IMG_LENGTH, IMG_WIDTH
# Convolution Neural Network


current_directory = os.path.dirname(os.path.abspath(__file__))
current_os = platform
if (current_os == 'win32'):
    MODEL_PATH = current_directory + '\models\model.h5'
    models_directory = current_directory + '\models\\'

else:
    MODEL_PATH = current_directory + '/models/model.h5'
    models_directory = current_directory + '/models/'

class DataSet(Preprocessing):

    def __init__(self):
        self.df = self.build_df() #For df
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
    
    def build_df(self):
        """Returns a pandas DataFrame of the image itself (matrix form) and the
        corresponding label"""

        X, y = self.get_images_and_labels()
        images_4d = X
        # Shape of np array is (30, 600, 600, 3)
        # We've gotta throw each image (600,600,3) into a single 1-D list so we can
        # convert it into a Series pandas object
        series_X = pd.Series( (v[0] for v in images_4d), name='image' )
        series_y = pd.Series(y, name='label')
        df = pd.concat([series_X, series_y], axis=1)
        return df


    def build_dataset(self):
        # df related
        # X = self.df.iloc[:, :self.df.shape[1]-1] #images
        # y = self.df['label'] #label
        X, y = self.get_images_and_labels()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random.randint(0, 100))
        print(X_train[0])
        print("--------------")
        print(y_train[0])
        print("--------------")
        print(X_test[0])
        print("--------------")
        print(y_test[0])
        print("--------------")
        X_valid, X_test, y_valid, y_test = train_test_split(X, y, test_size=0.5, random_state=random.randint(0, 100))

        # Squash all pixel values to values between 0 and 1 (inclusive)
        X_train /= 255.0
        X_valid /= 255.0
        X_test /= 255.0

        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')

        
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test


class Model(object):
    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.model = None
    
    def build_model(self, data_set):
        data_set.build_dataset()

        self.X_train = data_set.X_train
        self.X_valid = data_set.X_valid
        self.X_test = data_set.X_test
        self.y_train = data_set.y_train
        self.y_valid = data_set.y_valid
        self.y_test = data_set.y_test

        num_classes = data_set.df.shape[1]-1
        num_classes_one_hot = num_classes*2
        num_rows = self.X_train.shape[0]

        print(self.y_train)
        print(self.y_test)
        print(self.y_valid)

        # print ("INPUT SHAPE IS: %d, %d, %d" % (IMG_LENGTH, IMG_WIDTH, IMG_CHANNEL))
        # self.model = Sequential()
        # self.model.add(Conv2D(32, kernel_size = (5, 5),strides = (1,1),activation='relu', input_shape = (IMG_WIDTH, IMG_LENGTH, IMG_CHANNEL), data_format='channels_last'))
        # self.model.add(MaxPooling2D(64, (5, 5)))
        # self.model.add(Conv2D(64, (5,5), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2,2)))
        # self.model.add(Flatten(data_format='channels_last'))
        # self.model.add(Dense(1000, activation='relu'))
        # #self.model.add(Dense(num_classes_one_hot, activation='softmax'))
        # self.model.add(Dense(num_classes_one_hot, activation='softmax'))

        self.model = Sequential()

        self.model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=data_set.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes_one_hot))
        self.model.add(Activation('softmax'))

        # If models directory is empty, train and create a new model
        if not listdir(models_directory):
            self.train(batch_size=16, nb_epoch=5)
            self.model.summary() #Only call this after fitting the data
        else:
            self.load(MODEL_PATH)

    
        
    def train(self, batch_size=32, nb_epoch=40, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.y_test = to_categorical(self.y_test)
        self.y_train = to_categorical(self.y_train)
        self.y_valid = to_categorical(self.y_valid)

        print(self.y_test)
        print(self.y_train)
        print(self.y_valid)
        if (data_augmentation != True):
            print("Data augmentation is not true.")
            self.model.fit(self.X_train, self.y_train, epochs=nb_epoch, batch_size = batch_size, shuffle=True)
        else:
            print("Commencing data augmentation...")
            # this will do preprocessing and realtime data augmentation
            data_generator = ImageDataGenerator(
                featurewise_center=False,             # set input mean to 0 over the dataset
                samplewise_center=False,              # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,   # divide each input by its std
                zca_whitening=False,                  # apply ZCA whitening
                rotation_range=20,                     # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,                # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,               # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,                 # randomly flip images
                vertical_flip=False)                  # randomly flip images
                        
            data_generator.fit(self.X_train)
            # self.y_test = to_categorical(self.y_test)
            # self.y_train = to_categorical(self.y_train)
            # self.y_valid = to_categorical(self.y_valid)

            self.model.fit_generator(data_generator.flow(self.X_train, self.y_train, batch_size=batch_size), steps_per_epoch=self.X_train.shape[0], epochs=nb_epoch, validation_data=(self.X_valid, self.y_valid))
            score = self.model.evaluate(self.X_test, self.y_test, batch_size=batch_size)
            print("The score is: ")
            print(score)    
    def save(self, file_path=MODEL_PATH):
        print("Saving model...")
        self.model.save(file_path)
        print("Successfully saved model.")

    def load(self, file_path=MODEL_PATH):
        print("Loading model...")
        self.model = keras.models.load_model(file_path)
        print("Successfully loaded model.")
    
    def predict(self, image):
        # Squash the image pixel values to between 0 and 1 (inclusive)
        image = image/255.0
        probability = self.model.predict(image)
        prediction = self.model.predict_classes(image)[0]
        print('Probability is: ', probability)
        print('Prediction is: ', prediction)
        return prediction






