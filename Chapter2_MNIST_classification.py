# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 20:31:55 2022

@author: damon
"""

# Just MNIST

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#from matplotlib import pyplot as plt
#plt.imshow(test_images[2])

from keras import models
from keras import layers
net = models.Sequential()
net.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
net.add(layers.Dense(10, activation='softmax'))
net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28)).astype('float32')/255
test_images = test_images.reshape((10000, 28*28)).astype('float32')/255

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

net.fit(train_images, 
        train_labels,
        epochs=5,
        batch_size=128)

test_loss, test_acc = net.evaluate(test_images, test_labels)
print('test_loss', test_loss)
print('test_acc', test_acc)
'''
test_loss 0.06335888803005219
test_acc 0.9799000024795532
'''