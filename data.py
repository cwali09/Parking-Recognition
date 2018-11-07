from os import chdir, listdir
from os.path import isfile, join
import pandas as pd

class DataRetrieval(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.images = None
        self.labels = None

    
    def retrieve_from_images(self):
        pass
    

    """Returns tuple with the images URL and labels associated with each image. (Correct, Incorrect)"""
    def get_image_label_df(self, training_data="./data"):
        #chdir(path=training_data)
        images_path = ''
        images = [f for f in listdir(training_data) if isfile(join(images_path, f))]
        labels = [item for item in images if item.endswith(".jpg")]
        return (images, labels)