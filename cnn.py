from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten

from data_retrieval import DataRetrieval
# Convolution Neural Network

class Model(object):
    def __init__(self):
        self.df = DataRetrieval().get_image_label_df()
        # Get number of columns (attributes)
        num_classes = self.df.shape[1]

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size = (5, 5),strides = (1,1),activation='relu'))
        self.model.add(MaxPooling2D(64, (5, 5)))
        self.model.add(Conv2D(64, (5,5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten(data_format='channels_last'))
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
    
    def train(self):
        pass
        
a = Model()






