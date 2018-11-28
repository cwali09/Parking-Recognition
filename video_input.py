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
        incorrect_parking_counter = 0
        while True:
                        
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
                incorrect_parking_counter += 1
                blink()
                if incorrect_parking_counter >= 15 and is_same_misparked_car != True:
                    print("Sending email...")
                    send_email()
                    is_same_misparked_car = True
                    incorrect_parking_counter = 0
            else:
                is_same_misparked_car = False
                incorrect_parking_counter = 0




            



