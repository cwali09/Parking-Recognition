from os import chdir, listdir
import os.path
import pandas as pd
import numpy as np

# DataRetrieval is responsible for getting all the image URLs and labels (correct or incorrect)
# and organizing them into a single Pandas DataFrame (dict_of_lists_to_df and get_image_label_df). 
# Once the dataframe has been made then it uses split_to_attribute_set_and_class_label
# to return a tuple of pandas series objects (the attribute set, the target class) so 
# the Preprocessing class can work its magic.

class DataRetrieval(object):

    def __init__(self):
        self.df = self.get_image_label_df(self)
        self.images, self.labels = self.split_to_attribute_set_and_class_label()


    """Turn a dictionary of lists into a Pandas Dataframe"""
    def dict_of_lists_to_df(self, dict_of_lists):
        return pd.DataFrame.from_dict(dict_of_lists)
    
    """Return a Dataframe with each image's URL, and if they're correctly parked (Correct=1, Incorrect=0)"""
    def get_image_label_df(self, training_data="./data"):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        image_label = {'image': [], 'label': []}

        # Get all images in correct folder
        correct_directory = current_directory + '\data\correct\\'
        incorrect_directory = current_directory + '\data\incorrect\\'

        for image in listdir(correct_directory):
            if (image.endswith(".JPG")):
                image_label['image'].append(correct_directory + image)
                image_label['label'].append(1)
        # Get all images in incorrect folder
        for image in listdir(incorrect_directory):
            if (image.endswith(".JPG")):
                image_label['image'].append(incorrect_directory + image)
                image_label['label'].append(0)
        # Transform dictionary of images and labels to a Pandas Dataframe
        df = self.dict_of_lists_to_df(image_label)
        return df
    
    def split_to_attribute_set_and_class_label(self):
        """Splits the pandas DataFrame into 2 pandas series: the attribute set (X) and class label (y)"""
        df = self.get_image_label_df()
        X = df.iloc[:, :df.shape[1]-1].values
        y = df['label'].values
        return (X, y)

    # def get_split(self):
    #     return self.split_to_attribute_set_and_class_label()
    
    # def get_df(self):
    #     return self.get_image_label_df()

a = DataRetrieval()
#print(a.get_image_label_df())
# print(a.split_to_attribute_set_and_class_label()[0])