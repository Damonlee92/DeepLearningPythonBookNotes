# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:06:04 2022

@author: damon
"""
import numpy as np

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index() # to decode the features
# For windows, data stored in C:\Users\damon\.keras\datasets

#print("train_data[0] is:")
#print(train_data[0])

#print("train_labels[0] is:")
#print(train_labels[0])

#print("max of max, see max val.")
#print(max([max(sequence) for sequence in train_data]))

def decodeFeatures(row=0):
    """
    View one of the imdb reviews
    """
    reverse_word_index = dict([
        (value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join(
        [reverse_word_index.get(i-3, '?') for i in train_data[row]])
    return decoded_review

#print(decodeFeatures(1)) # try view any review

def vectorizeSequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
    
x_train = vectorizeSequences(train_data)
x_test = vectorizeSequences(test_data)

#print("x_train[0]")
#print(x_train[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Make the NN
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation=('sigmoid')))


#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])

from keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])


x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, 
                    partial_y_train, 
                    epochs=20, #20 overfits. use 4.
                    batch_size=512, 
                    validation_data=(x_val, y_val))

historyD = history.history
print("History results:", historyD.keys)

import matplotlib.pyplot as plt
epochs = range(1, len(historyD['loss']) + 1)
plt.plot(epochs, historyD['loss'], 'bo', label='training loss')
plt.plot(epochs, historyD['val_loss'], 'b', label='validation loss')
plt.ylim([0, 1])
plt.title('traing & validaiton loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()
epochs = range(1, len(historyD['binary_accuracy']) + 1)
plt.plot(epochs, historyD['binary_accuracy'], 'ro', label='training accuracy')
plt.plot(epochs, historyD['val_binary_accuracy'], 'r', label='validation accuracy')
plt.ylim([0, 1])
plt.title('traing & validaiton accuracy')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#results=model.evaluate(x_test, y_test)
#print(results)
#print("epochs=4 results are [0.6931695342063904, 0.5]")


model.predict(x_test)







