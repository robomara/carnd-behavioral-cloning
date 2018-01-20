# import packages
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import keras specific packages
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# read in csv of file locations
lines = []
with open('./mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# process images and measurements
images = []
measurements = []

for line in lines:
    # steering measurement
    steerCenter = float(line[3])
    
     # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steerLeft = steerCenter + correction
    steerRight = steerCenter - correction
    
    # load images for center, left and right
    center = './mydata/IMG/' + line[0].split("\\")[-1]
    left = './mydata/IMG/' + line[1].split("\\")[-1]
    right = './mydata/IMG/' + line[2].split("\\")[-1]
    
    imgCenter = cv2.imread(center)
    imgLeft = cv2.imread(left)
    imgRight = cv2.imread(right)
    
    # create mirrored center image to remove left bias
    steerCenterRev = steerCenter*-1.0
    imgCenterRev = cv2.flip(imgCenter,1)
    
    images.extend([imgCenter,imgLeft,imgRight,imgCenterRev])
    measurements.extend([steerCenter,steerLeft,steerRight,steerCenterRev])
    



# convert measurements and images to numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)


# begin model
# NOTES: relu introduces non=linearites. dropout reduces overfitting. Validation of 20% used.
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))) # crop image to remove sky and car body
model.add(Lambda(lambda x: x/255.0)) # normalise
model.add(Convolution2D(5,5,5,activation="relu")) 
model.add(MaxPooling2D())
model.add(Convolution2D(5,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(5,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(5,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(p=0.5)) # reduce overfitting with dropout
model.add(Flatten())
model.add(Dense(600))
model.add(Dense(300))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,nb_epoch=10,validation_split=0.2, shuffle=True)

model.save('model.h5')
