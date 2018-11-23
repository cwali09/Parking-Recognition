import picamera
import time
import numpy as np

from cnn import Model
from CONSTANTS import IMG_CHANNEL, IMG_LENGTH, IMG_WIDTH

def stream(model):
    while True:
        with picamera.PiCamera() as camera:
            camera.resolution = (IMG_WIDTH, IMG_LENGTH)
            camera.framerate = 24
            
            image = np.empty((IMG_WIDTH, IMG_LENGTH, IMG_CHANNEL), dtype=np.uint8)
            #image = np.empty((128, 128, IMG_CHANNEL,), dtype=np.uint8)
            camera.start_preview()
            camera.capture(image, 'rgb')
            #time.sleep(2)
            camera.stop_preview()
            
            print(image.shape)
            #image = cv2.resize(image, (IMG_LENGTH, IMG_WIDTH))
            image = image.reshape(1,IMG_WIDTH,IMG_LENGTH, IMG_CHANNEL)
            #image = image.reshape(1, 100, 100, 3)
            prediction = model.predict(image)



            



