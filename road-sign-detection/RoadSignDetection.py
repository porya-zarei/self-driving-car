import math
import tensorflow as tf
# from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# import random
# import pickle
# import pandas as pd
import cv2
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os

class RoadSignDetector():
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        self.classes = ['Speed limit(20km/h)', 'Speed limit(30km/h)', 'Speed limit(50km/h)', 'Speed limit(60km/h)', 'Speed limit(70km/h)', 'Speed limit(80km/h)', 'End of speed limit(80km/h)', 'Speed limit(100km/h)', 'Speed limit(120km/h)', 'No passing', 'No passing for vechiles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vechiles', 'Vechiles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
                        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vechiles over 3.5 metric tons,']

    def load_model(self):
        # model = self.__modified_model()
        # model.load_weights(self.model_path)
        self.model = keras.models.load_model(self.model_path)
        # print("Model loaded",self.model.summary())
        # model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy',metrics=['accuracy'])
        # return model

    def __modified_model(self):
        model = Sequential()
        model.add(Conv2D(60, (5, 5), input_shape=(
            32, 32, 1), activation='relu'))
        model.add(Conv2D(60, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(30, (3, 3), activation='relu'))
        model.add(Conv2D(30, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(43, activation='softmax'))

        model.compile(Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def grayscale(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', img)
        return img

    def equalize(self, img):
        img = cv2.equalizeHist(img)
        # cv2.imshow('equalized', img)
        return img

    def preprocess(self, img):
        img = self.grayscale(img)
        img = self.equalize(img)
        img = img/255
        # cv2.imshow('preprocessed', img)
        return img

    def resize(self, image):
        image = cv2.resize(image, (32, 32))
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        image = self.preprocess(image)
        image = self.resize(image)
        # cv2.imshow('resized', image)
        return self.model.predict(image.reshape(1, 32, 32, 1))

    def predict_classes(self, image):
        pred = self.predict(image)
        out_pred = np.round(pred, 2).astype(float)[0]
        max_index = np.argmax(out_pred)
        self.title = self.classes[max_index]
        self.probability = float(math.ceil(out_pred[max_index]*1000)/1000)
        return (self.title, self.probability)

    def get_texted_image(self, image, point=(10, 30), color=(0, 255, 0), thickness=2):
        img = cv2.putText(image, self.title, point,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        img = cv2.putText(img, str(self.probability), (point[0], point[1]+30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        return img


detector = RoadSignDetector('./resources/signs/signs_model.h5')
img = cv2.imread('./resources/speed-limit-60.png')
detector.predict_classes(img)
cv2.imshow('Result', detector.get_texted_image(img))
cv2.waitKey(0)