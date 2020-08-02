"""Training a neural network using keras and the MNIST dataset"""
import gzip
import os
import pickle
from urllib import request

import keras
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential


def fixlabels(labels):
    """Changes the labels to be used in the keras model."""
    new_labels = []
    for label in labels:
        tmp_arr = [0] * 10
        tmp_arr[label] = 1

        new_labels.append(np.array(tmp_arr))

    return np.array(new_labels)


np.random.seed(1234)

BATCH_SIZE = 1024
NUM_CLASSES = 10
EPOCHS = 20

# input image dimensions
img_rows, img_cols = 28, 28

# ===============================
# Loading the MNIST dataset
# ===============================
url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
if not os.path.isfile("mnist.pkl.gz"):
    request.urlretrieve(url, "mnist.pkl.gz")

data_file = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(data_file, encoding='latin1')
data_file.close()

# ===============================
# Creating a model
# ===============================
model = Sequential()
# Input layer
model.add(Dense(196, activation='relu', input_shape=(img_rows * img_cols,)))

# Hidden layers
model.add(Dense(49, activation='relu'))
model.add(Dropout(0.5))  # Helps preventing overfitting

# Output layer
# Softmax chooses the most prominent answer
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compiling model
# Adam climbs to a higher accuracy much quicker than SGD So I'm using this one
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])

# ===============================
# Train the model
# ===============================
model.fit(train_set[0], fixlabels(train_set[1]),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(valid_set[0], fixlabels(valid_set[1])))

# ===============================
# print the scores
# ===============================
score = model.evaluate(test_set[0], fixlabels(test_set[1]), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
