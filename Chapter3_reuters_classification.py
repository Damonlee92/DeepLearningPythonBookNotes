# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:06:04 2022

@author: damon
"""
import numpy as np
# For windows, data stored in C:\Users\damon\.keras\datasets
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def decodeNewswire(row=0):
    word_index = reuters.get_word_index()
    reverse_word_index = dict([
        (value, key) for (key, value) in word_index.items()
        ])
    decoded_newswire = ' '.join([reverse_word_index.get(i-3, '???') for i in train_data[row]])
    return decoded_newswire

def vectorizeSequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
    
x_train = vectorizeSequences(train_data)
x_test = vectorizeSequences(test_data)

from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Make the NN
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation=('softmax')))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, 
                    partial_y_train, 
                    epochs=20, #20 overfits. use 4.
                    batch_size=512, 
                    validation_data=(x_val, y_val))

historyD = history.history
print("History results:", historyD.keys)

import matplotlib.pyplot as plt
epochs = range(1, len(historyD['loss']) + 1)
plt.plot(epochs, historyD['loss'], 'go', label='training loss')
plt.plot(epochs, historyD['val_loss'], 'g', label='validation loss')
plt.title('traing & validaiton loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()
epochs = range(1, len(historyD['accuracy']) + 1)
plt.plot(epochs, historyD['accuracy'], 'bo', label='training accuracy')
plt.plot(epochs, historyD['val_accuracy'], 'b', label='validation accuracy')
plt.title('traing & validaiton accuracy')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

