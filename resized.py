from os import listdir
from cnn import current_directory
import cv2

incorrect64_directory = current_directory+'\incorrect-64\\'
correct64_directory = current_directory + '\correct-64\\'
validate64_directory = current_directory + '\\validate-64\\'

#correct_directory = current_directory + '\data\correct\\'
correct_directory = "C:\\Users\\cwali\\OneDrive\\Desktop\\Parking-Recognition\\data\\correct\\"
incorrect_directory = current_directory + '\data\incorrect\\'
validate_directory = current_directory + '\data\\validate\\'
count = 1

for image_url in listdir(validate_directory):
    image_url = validate_directory+image_url
    image = cv2.imread(image_url)

    print(image_url)
    print(image)
    image = cv2.resize(image, (64, 64))
    cv2.imwrite(validate64_directory+'validate-image%d.jpg' % count, image)
    count = count + 1