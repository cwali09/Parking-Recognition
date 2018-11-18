import PiCamera
import time
import numpy as np

from cnn import Model

def stream(model):
    while True:
        with picamera.PiCamera() as camera:
            camera.resolution(600, 600)
            camera.framerate = 24
            image = np.empty((600, 600, 3), dtype=float)
            camera.capture(output, 'rgb')

            image.reshape(1,600,600,3)
            prediction = model.predict(image)


            



