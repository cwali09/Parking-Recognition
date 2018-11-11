from data_retrieval import DataRetrieval
import pandas as pd
import numpy as np
import cv2

class Preprocessing(DataRetrieval):
    # Is it better to use self.X and self.y in crop_images directly, or should I pass them
    # in as arguments in the constructor?

    def __init__(self):
        self.X, self.y = DataRetrieval().split_to_attribute_set_and_class_label()
        self.X = self.crop_images(self.X)

    def crop_images(self, X):
        cropped_images = []

        # Iterating through series X
        for contained_url in X:
            for image_url in contained_url:
                # Each image is 3 dimensional. (Sample shape would be like (600, 600, 3))
                img = cv2.imread(image_url)
                #scaled_y, scaled_x = int(img.shape[0]*(1/6)), int(img.shape[1]*(1/4))
                #cropped_img = cv2.resize(img, (scaled_x, scaled_y))
                cropped_img = cv2.resize(img, (600, 600))
                cropped_images.append(cropped_img)
        return cropped_images

a = Preprocessing()
#print(a.X)
