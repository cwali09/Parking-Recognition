from data_retrieval import DataRetrieval
import pandas as pd
import numpy as np
import cv2

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
        #images = np.array(images)
        cropped_images = []

        # Iterating through series X
        for image in images:
            # Each image is 3 dimensional. (Sample shape would be like (600, 600, 3))
            cropped_img = cv2.resize(image, (600, 600))
            cropped_images.append(cropped_img)
            #cropped_images = np.append(cropped_images, cropped_img)
        cropped_images = np.array(cropped_images)
        return cropped_images
    
    def get_images_and_labels(self):
        #return self.X, self.y
        X = self.crop_images(self.get_images_from_url(DataRetrieval().images))
        y = DataRetrieval().labels
        return X, y
    
    def enable_edge_detection(self, images):
        pass

a = Preprocessing()
#print(a.X.shape)
#print(a.X)
