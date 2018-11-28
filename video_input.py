import picamera
import time
import numpy as np
import cv2

from cnn import Model, current_directory
from led_blink import blink
from email_sender import send_email
from CONSTANTS import IMG_CHANNEL, IMG_LENGTH, IMG_WIDTH

np.set_printoptions(threshold=np.nan)

def stream(model):
    with picamera.PiCamera() as camera:
        camera.resolution = (IMG_WIDTH, IMG_LENGTH)
        x, y, w, h = 0.20, 0.37, 0.59, 0.8
        camera.zoom = x, y, w, h
        camera.rotation = 180
        camera.start_preview()
        sleep_counter = 0
        while True:
            sleep_counter += 1
                        
            image = np.empty((IMG_WIDTH, IMG_LENGTH, IMG_CHANNEL), dtype=np.uint8)
            camera.capture('image.jpg')
            image = cv2.imread('./image.jpg')
            #camera.capture(image, 'rgb')
            #camera.stop_preview()
            
            #print(image.shape)
            image = cv2.resize(image, (IMG_LENGTH, IMG_WIDTH))
            image = image.reshape(1,IMG_WIDTH,IMG_LENGTH, IMG_CHANNEL)
            #print(image)
            prediction = model.predict(image)
            if (prediction == 0):
                blink()
                if sleep_counter >= 30:
                    send_email()
                    sleep_counter = 0




            



