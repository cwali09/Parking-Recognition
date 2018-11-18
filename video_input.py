import picamera
import time
import numpy as np

from cnn import Model
from CONSTANTS import IMG_CHANNEL, IMG_LENGTH, IMG_WIDTH

def stream(model):
    while True:
        with picamera.PiCamera() as camera:
            camera.resolution(640, 480)
            camera.framerate = 80
            image = np.empty((IMG_LENGTH, IMG_WIDTH, IMG_CHANNEL), dtype=float)
            camera.capture(output, 'rgb')

            image.reshape(1,IMG_LENGTH,IMG_WIDTH, IMG_CHANNEL)
            prediction = model.predict(image)



            



