import os
import cv2
import numpy
from random import shuffle

from CONSTANTS import IMG_LENGTH, IMG_WIDTH, IMG_CHANNEL

numpy.set_printoptions(threshold=numpy.nan)

def test_validation(model):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    validation_directory = current_directory + '/data/validate/'
    shuffled_validation_file_list = os.listdir(validation_directory)
    shuffle(shuffled_validation_file_list)

    for image_name in shuffled_validation_file_list:
        if (image_name.endswith(".jpg") or image_name.endswith(".JPG")):
            image = cv2.imread(validation_directory + image_name)
            resized_img = cv2.resize(image, (IMG_LENGTH, IMG_WIDTH))
            resized_img = resized_img.reshape(1, IMG_LENGTH, IMG_WIDTH, IMG_CHANNEL)
            print("\nPredicting ", image_name)
            print(resized_img)
            model.predict(resized_img)