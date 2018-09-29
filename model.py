import csv
import cv2
import numpy as np
import os
import sklearn
import random
from PIL import Image
from sklearn.model_selection import train_test_split

def process_image(img):
    return img

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

random.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            car_images = []
            steering_angles = []
            path = 'data/IMG/'
            for sample in batch_samples:
                # create adjusted steering measurements for the side camera images
                correction = 0.2
                steering_center = float(sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                img_center = process_image(np.asarray(Image.open(path + os.path.basename(sample[0])).convert("RGB")))
                img_left = process_image(np.asarray(Image.open(path + os.path.basename(sample[1])).convert("RGB")))
                img_right = process_image(np.asarray(Image.open(path + os.path.basename(sample[2])).convert("RGB")))

                # add images and angles to data set
                car_images.extend([img_center, img_left, img_right])
                steering_angles.extend([steering_center, steering_left, steering_right])

                # augmented
                car_images.extend([cv2.flip(img_center, 1), cv2.flip(img_left, 1), cv2.flip(img_right, 1)])
                steering_angles.extend([steering_center * (-1.0), steering_left * (-1.0), steering_right * (-1.0)])
            
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

#NVIDIA
model.add(Convolution2D(24, 5, 5, subsample=(2, 2),  activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2),  activation = "relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2),  activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss="mse", optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch = len(train_samples),
                    validation_data = validation_generator,
                    validation_steps = len(validation_samples),
                    epochs = 2,
                    verbose = 1)

model.save('model.h5')
exit()
 