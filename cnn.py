import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from data_retrieval import DataRetrieval
from preprocessing import Preprocessing
# Convolution Neural Network

class DataSet(Preprocessing):
    def __init__(self):
        self.df = self.build_df()
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
        X = self.df.iloc[:, :self.df.shape[1]-1] #images
        y = self.df['label'] #labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random.randint(0, 100))
        X_valid, X_test, y_valid, y_test = train_test_split(X, y, test_size=0.5, random_state=random.randint(0, 100))

        # Squash all pixel values to values between 0 and 1 (inclusive)
        X_train /= 255
        X_valid /= 255
        X_test /= 255

        print('X_train shape:', X_train.shape)
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
    def __init__(self, data_set):
        data_set.build_dataset()
        self.X_train = data_set.X_train
        self.X_valid = data_set.X_valid
        self.X_test = data_set.X_test
        self.y_train = data_set.y_train
        self.y_valid = data_set.y_valid
        self.y_test = data_set.y_test

        # Get number of columns (attributes)
        num_classes = data_set.df.shape[1]

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size = (5, 5),strides = (1,1),activation='relu'))
        self.model.add(MaxPooling2D(64, (5, 5)))
        self.model.add(Conv2D(64, (5,5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten(data_format='channels_last'))
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.summary() #Only call this after fitting the data
    
        
    def train(self):
        pass                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

a = DataSet()
a.build_dataset()
# df = DataRetrieval().get_df()        
a = Model(a)






