from os import chdir, listdir
import os.path
import pandas as pd
import numpy as np

class DataRetrieval(object):

    def __init__(self):
        self.images = None
        self.labels = None

    """Turn a dictionary of lists into a Pandas Dataframe"""
    def dict_of_lists_to_df(self, dict_of_lists):
        return pd.DataFrame.from_dict(dict_of_lists)

    """Return a Dataframe with each image's URL, and if they're correctly parked (Correct=1, Incorrect=0)"""
    def get_image_label_df(self, training_data="./data"):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        image_label = {'image': [], 'label': []}

        # Get all images in correct folder
        for image in listdir(current_directory + '/data/correct'):
            if (image.endswith(".JPG")):
                image_label['image'].append(image)
                image_label['label'].append(1)
        # Get all images in incorrect folder
        for image in listdir(current_directory + '/data/incorrect'):
            if (image.endswith(".JPG")):
                image_label['image'].append(image)
                image_label['label'].append(0)
        # Transform dictionary of images and labels to a Pandas Dataframe
        df = self.dict_of_lists_to_df(image_label)
        return df
#a = DataRetrieval()
#print(a.get_image_label_df())