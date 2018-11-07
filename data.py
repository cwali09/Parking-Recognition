from os import chdir, listdir
from os.path import isfile, join

class DataRetrieval(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.images = None
        self.labels = None

    def retrieve_from_images(self):
        pass
    

    """Changes the current working directory to that of where the images to train on are"""
    def traverse_to_image_directory(self):
        training_data = "./data" # Climb to training data directory
        chdir(path=training_data)
        pass