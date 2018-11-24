from os import chdir, listdir, path, makedirs
from sys import platform

from data_retrieval import DataRetrieval
from preprocessing import Preprocessing
from cnn import DataSet, Model, current_directory, models_directory, current_os, MODEL_PATH
from test_against_validation import test_validation

from keras.models import load_model
import keras.models

from CONSTANTS import IMG_LENGTH, IMG_WIDTH, IMG_CHANNEL
import cv2

def check_models_directory_exists(models_path=models_directory):
    if not path.exists(models_directory):
        print("Models directory does not exist. Attempting to create...")
        makedirs(models_directory)
        if path.exists(models_directory):
            print("Successfully created Models directory.")
        else:
            print("Could not create Models directory. ABORT")
            exit()




check_models_directory_exists()
model = Model()

if (current_os == "win32"):
    if not listdir(models_directory):
        data_set = DataSet()
        data_set.build_dataset()
        model.build_model(data_set)
        model.save()
    else:
        # pass
        model.load(file_path = MODEL_PATH)
        test_validation(model)
    # data_set = DataSet()
    # data_set.build_dataset()
    # model.build_model(data_set)
    # model.save()
else:
    # For Linux
    from video_input import stream
    from test_against_validation_linux import test_validation

    
    model.load(file_path=MODEL_PATH)
    #test_validation(model)
    stream(model)
    


