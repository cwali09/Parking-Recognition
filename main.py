from os import chdir, listdir, path
from sys import platform

from data_retrieval import DataRetrieval
from preprocessing import Preprocessing
from cnn import DataSet, Model, current_directory, models_directory, current_os, MODEL_PATH
from keras.models import load_model
import keras.models


model = Model()

if (current_os == "win32"):
    if not listdir(models_directory):
        data_set = DataSet()
        data_set.build_dataset()
        model.build_model(data_set)
        model.save()
    else:
        model.load(file_path = MODEL_PATH)