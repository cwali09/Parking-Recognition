from picamera import PiCamera
from time import sleep
from os import listdir
from os import rename

#size = len(listdir('./'))
#images = listdir('./')
#for image in images:
#    imageSource = image
#    image = image.split('-')
    #print(image)
#    if (image[0] == 'incorrect'):
#        image[0] = 'correct'
#        imageDest = image[0]+'-'+image[1]
#        rename(imageSource, imageDest)
#        
#sleep(100)

camera = PiCamera()

camera.rotation = 180
camera.start_preview()


#x, y, w, h = 0.20, 0.33, 0.59, 0.46
x, y, w, h = 0.20, 0.37, 0.59, 0.8

camera.resolution = (64, 64)
camera.zoom = x, y, w, h

size = len(listdir('./'))
print(size)

#sleep(100)
#while True:
#    camera.capture('zzzz.jpg')

for x in range(400):
    camera.capture('empty-image%s.jpg' % (x+size))
camera.stop_preview()
