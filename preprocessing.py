from data_retrieval import DataRetrieval
from CONSTANTS import IMG_CHANNEL, IMG_LENGTH, IMG_WIDTH

import pandas as pd
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator

class Preprocessing(DataRetrieval):
    # Is it better to use self.X and self.y in crop_images directly, or should I pass them
    # in as arguments in the constructor?

    def __init__(self):
        self.X, self.y = DataRetrieval().images, DataRetrieval().labels
        # X is now a matrix of images rather than matrix of URLs
        self.X = self.crop_images(self.get_images_from_url(self.X))
    
    def get_images_from_url(self, X):
        images = []

        for contained_url in X:
            for image_url in contained_url:
                # Each image is 3 dimensional. (Sample shape would be like (600, 600, 3))
                img = cv2.imread(image_url)
                images.append(img)
        return images

    def crop_images(self, images):
        """"Returns an array of cropped images (600x600x3) from inputted Pandas series of URLs"""
        cropped_images = []

        # Iterating through series X
        for image in images:
            # Each image is 3 dimensional. (Sample shape would be like (600, 600, 3)) -- (x,y,z)
            cropped_img = cv2.resize(image, (IMG_WIDTH, IMG_LENGTH))
            cv2.imshow('hello', cropped_img)
            cv2.waitKey(3)
            cropped_images.append(cropped_img)
        cropped_images = np.array(cropped_images, dtype='float')
        return cropped_images
    
    
    def process(self, X_train):
        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
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
        datagen.fit(X_train)
        return X_train, datagen
    
    def get_images_and_labels(self):
        #return self.X, self.y
        X = self.crop_images(self.get_images_from_url(DataRetrieval().images))
        y = DataRetrieval().labels
        return X, y
