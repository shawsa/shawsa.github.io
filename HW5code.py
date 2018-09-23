# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

def plot_im(x):
    x=x.reshape((28, 28))
    plt.imshow(x[::-1],cmap='BuPu', origin='higher')
    plt.show()

# reading the MNIST data
file = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = cPickle.load(file)
file.close()

# converting training data into a vector format
tr_v = [np.reshape(training_digit, (784, 1)) for training_digit in training_data[0]]
# saving the actual digit information in different name
tr_d = training_data[1]

separated = [[],[],[],[],[],[],[],[],[],[]]
for i, vec in zip(tr_d, tr_v):
    separated[i].append(vec)
            

# creating an empty matrix
average = np.zeros((10,784))
    
#average = [[],[],[],[],[],[],[],[],[],[]]
for digit in xrange(0,10):
    average[digit] = np.mean(np.stack(separated[digit]),axis=0).reshape(784)
    #average[digit] = np.mean(np.stack(separated[digit]),axis=0)
    
plt.plot(np.reshape(average[2], (28, 28)),'o')
plt.show()

#plt.imshow(np.reshape(average[6], (28, 28)),cmap='BuPu', origin='higher')
plot_im(average[0])