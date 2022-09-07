# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:15:41 2022

@author: I561078
"""
import os
from sklearn.datasets import load_files
import numpy as np
from numpy import random
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import img_to_array, array_to_img, load_img
from PIL import Image,ImageOps
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import np_utils
import pathlib
from datetime import datetime
from ai_core_sdk.models import Metric, MetricTag, MetricCustomInfo, MetricLabel
from ai_core_sdk.tracking import Tracking
import tensorflow as tf
import cv2

aic_connection = Tracking()

#
# Variables
TRAINING_PATH = '/app/data/archive/fruits-360_dataset/fruits-360/Training/'
TESTING_PATH = '/app/data/archive/fruits-360_dataset/fruits-360/Test'
#IMG_PATH = '/app/data/data/archive/fruits-360_dataset/fruits-360/Test2/images/'
MODEL_PATH = '/app/model/model.h5'

# Load Datasets
train_dir = pathlib.Path(TRAINING_PATH)
test_dir = pathlib.Path(TESTING_PATH)
n_classes = 2
X_train = []
y_train = []
X_test = []
y_test = []

n_train_apples_jpg = len(list(train_dir.glob("Apple/*.jpg")))
n_train_apples_jpeg = len(list(train_dir.glob("Apple/*.jpeg")))
n_train_lemons_jpg = len(list(train_dir.glob("Lemon/*.jpg")))
n_train_lemons_jpeg = len(list(train_dir.glob("Lemon/*.jpeg")))

n_test_apples_jpg = len(list(test_dir.glob("Apple/*.jpg")))
n_test_apples_jpeg = len(list(test_dir.glob("Apple/*.jpeg")))
n_test_lemons_jpg = len(list(test_dir.glob("Lemon/*.jpg")))
n_test_lemons_jpeg = len(list(test_dir.glob("Lemon/*.jpeg")))

n_train_apples = n_train_apples_jpg + n_train_apples_jpeg
n_train_lemons = n_train_lemons_jpg + n_train_lemons_jpeg 
n_test_apples = n_test_apples_jpg + n_test_apples_jpeg 
n_test_lemons = n_test_lemons_jpg + n_test_lemons_jpeg

y_train_apples = [0 for i in range(n_train_apples)]
y_train_lemons = [1 for i in range(n_train_lemons)]
y_test_apples = [0 for i in range(n_test_apples)]
y_test_lemons = [1 for i in range(n_test_lemons)]

y_train_apples.extend(y_train_lemons)
y_train = np.asarray(y_train_apples)

y_test_apples.extend(y_test_lemons)
y_test = np.asarray(y_test_apples)

no_of_classes = len(np.unique(y_train))

y_train = np_utils.to_categorical(y_train,no_of_classes)
y_test = np_utils.to_categorical(y_test,no_of_classes)
#y_train[0] # Note that only one element has value 1(corresponding to its label) and others are 0.

def get_imgs(directory, controller=False):
    images = []
    for img in list(directory.glob("Apple/*.jpg")):
        path = str(img)
        read_img = np.asarray(Image.open(path).convert('L'))
        images.append(read_img)
        
    for img in list(directory.glob("Lemon/*.jpg")):
        path = str(img)
        read_img = np.asarray(Image.open(path).convert('L'))
        images.append(read_img)
        
    if controller:
        for img in list(directory.glob("Apple/*.jpeg")):
            path = str(img)
            read_img = np.asarray(Image.open(path).convert('L'))
            images.append(read_img)
        for img in list(directory.glob("Lemon/*.jpeg")):
            path = str(img)
            read_img = np.asarray(Image.open(path).convert('L'))
            images.append(read_img)
        
    return np.asarray(images)

def resize_imgs(img):
    dim = (96, 96)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

def apply_resize(array):
    n = array.shape[0]
    dim = (n, 96, 96)
    store = np.zeros(shape=dim)
    for i in range(n):
        curr_img = array[i]
        new_img = resize_imgs(curr_img)
        store[i] = new_img
    return store

X_train = get_imgs(train_dir, True)
X_test = get_imgs(test_dir, True)

X_train = apply_resize(X_train)
X_test = apply_resize(X_test)

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

aic_connection.log_metrics(
    metrics = [
        Metric(
            name= "X_train rows", value= float(X_train.shape[0]), timestamp=datetime.utcnow()),
    ]
)

aic_connection.log_metrics(
    metrics = [
        Metric(
            name= "X_test rows", value= float(X_test.shape[0]), timestamp=datetime.utcnow()),
    ]
)


IMG_SIZE = 96
def augment(image):
    image = image.reshape(1, 96, 96)
    image = tf.image.random_crop(image, size=[1, IMG_SIZE, IMG_SIZE])
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, 0.2, 0.8)
    image = np.array(image).reshape(IMG_SIZE, IMG_SIZE)
    return image

lemon_train = X_train[122:]
storage = np.zeros(shape=(100, 96, 96))
for i in range(100):
    random_img = random.randint(0, 8)
    test_img = lemon_train[random_img]
    augmented_img = augment(test_img)
    storage[i] = augmented_img
    
X_train = np.concatenate((X_train, storage), axis=0)
y_train2 = [1 for i in range(100)]
y_train2 = np_utils.to_categorical(y_train2,2)
y_train = np.concatenate((y_train, y_train2), axis=0)


#
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2,input_shape=(96,96,1),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())
#model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(2,activation = 'softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Train model
batch_size = 32
history = model.fit(X_train,y_train,
        batch_size = 32,
        epochs=15,
        verbose=2, shuffle=True)
#
# Test model
score = model.evaluate(X_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])
# Output will be available in logs of SAP AI Core.
# Not the ideal way of storing /reporting metrics in SAP AI Core, but that is not the focus this tutorial
aic_connection.log_metrics(
    metrics = [
        Metric(
            name= "Test Accuracy",
            value= score[1],
            timestamp=datetime.utcnow(),
            labels= [
                MetricLabel(name="metrics.ai.sap.com/Artifact.name", value="fruitmodel")
            ]
        )
    ]
)
#
# Save model
model.save(MODEL_PATH)
