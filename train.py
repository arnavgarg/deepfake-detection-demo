import numpy as np
import random
from classifiers import *
from pipeline import *
from datetime import datetime
from pathlib import Path
import json
import os
from tqdm import tqdm
import pickle

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print('Loading classifier...')
classifier = Meso4()
classifier.load('Meso4_DF.h5')
for layer in classifier.model.layers[:-4]:
    layer.trainable = False
classifier.model.compile(optimizer = Adam(learning_rate = 0.005), loss = 'mean_squared_error', metrics = ['accuracy'])
########################################
path = os.getcwd()

print('Loading train data...')
with open(str(path) + '/X_train.pickle', 'rb') as f:
    X_train = np.array(pickle.load(f))
with open(str(path) + '/y_train.pickle', 'rb') as f:
    y_train = np.array(pickle.load(f))

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=120,
    width_shift_range=0.5,
    height_shift_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.05,
    shear_range=120,
    zoom_range=0.9
)
datagen.fit(X_train)

print('Training model...')
classifier.model.fit(
    datagen.flow(X_train, y_train, batch_size=32, subset='training'),
    validation_data = datagen.flow(X_train, y_train, batch_size=8, subset='validation'),
    # steps_per_epoch = len(X_train) / 32, 
    epochs = 100
)
# classifier.model.fit(
#     x = X_train,
#     y = y_train,
#     epochs = 5,
#     verbose = 2,
#     shuffle = True
# )

name = './model_' + str(datetime.now()) + '.hd5'
print('Saving new weights to ' + name + '...')
classifier.model.save_weights(name)

print('Loading test data...')
with open(str(path) + '/X_test.pickle', 'rb') as f:
    X_test = np.array(pickle.load(f))
with open(str(path) + '/y_test.pickle', 'rb') as f:
    y_test = np.array(pickle.load(f))

print('Evaluating model...')
loss = classifier.model.evaluate(
    x = X_test,
    y = y_test
)

print('Done!')
