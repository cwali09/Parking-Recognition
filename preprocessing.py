from data_retrieval import DataRetrieval
from CONSTANTS import IMG_CHANNEL, IMG_LENGTH, IMG_WIDTH

import pandas as pd
import numpy as np
import cv2
from time import sleep
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
                # cv2.imshow('image', img)
                # sleep(10)
                images.append(img)
        return images

    # def crop_images(self, images):
    #     """"Returns an array of cropped images (600x600x3) from inputted Pandas series of URLs"""
    #     #images = np.array(images)
    #     cropped_images = []

    #     # Iterating through series X
    #     for image in images:
    #         print(image.shape)
    #         im_len, im_wid, im_chan = image.shape
            
    #         # Each image is 3 dimensional. (Sample shape would be like (600, 600, 3))
    #         width = IMG_WIDTH 
    #         height = IMG_LENGTH 
    #         inter = cv2.INTER_AREA
    #         # initialize the dimensions of the image to be resized and
    #         # grab the image size
    #         dim = None
    #         (h, w) = image.shape[:2]

    #         # if both the width and height are None, then return the
    #         # original image
    #         if width is None and height is None:
    #             return image

    #         # check to see if the width is None
    #         if width is None:
    #             # calculate the ratio of the height and construct the
    #             # dimensions
    #             r = height / float(h)
    #             dim = (int(w * r), height)

    #         # otherwise, the height is None
    #         else:
    #             # calculate the ratio of the width and construct the
    #             # dimensions
    #             r = width / float(w)
    #             dim = (width, int(h * r))

    #         # resize the image
    #         resized = cv2.resize(image, dim, interpolation = inter)

    #         # return the resized image
    #         cv2.imshow('image', resized)
    #         sleep(400)
    #         cropped_images.append(resized)

    #     cropped_images = np.array(cropped_images, dtype='float')
    #     return cropped_images
    
    def crop_images(self, images):
        """"Returns an array of cropped images (600x600x3) from inputted Pandas series of URLs"""
        #images = np.array(images)
        cropped_images = []

        # Iterating through series X
        for image in images:
            # print(image.shape)
            # im_len, im_wid, im_chan = image.shape
            
            # Each image is 3 dimensional. (Sample shape would be like (600, 600, 3))
            cropped_img = cv2.resize(image, (IMG_LENGTH, IMG_WIDTH))
            cropped_images.append(cropped_img)
            cv2.imwrite('CROPPED.jpg', cropped_img)
        cropped_images = np.array(cropped_images, dtype='float')
        return cropped_images
    
    def process(self, X_train):
        # this will do preprocessing and realtime data augmentation
        #X_train = pd.Series.as_matrix(X_train)
        print(X_train)
        print(X_train.shape)
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

#a = Preprocessing()
#print(a.X.shape)
#print(a.X)
