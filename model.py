from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop
from timeit import timeit
import time
import numpy as np

from format import *

SONG_NAME = "bebop.midi"
song_matrix = midi_to_matrix(SONG_NAME)


X = np.asarray([[x[0] for x in row] for row in song_matrix])
Y = np.asarray([[x[1] for x in row] for row in song_matrix])

n = int(len(X)*0.9)

Xtrain = X[:n]
Ytrain = Y[:n]
Xtest = X[n:]
Ytest = Y[n:]

print Xtrain.shape
print Xtest.shape

in_dim = len(Xtrain[0])

model = Sequential()
model.add(LSTM(128, input_shape=(None,in_dim)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

print "Compiling"
start = time.time()
model.compile(loss='mean_squared_error', optimizer=sgd)
elapsed = time.time() - start
print elapsed/60, ' and ', elapsed%60

# nb = 10
# model.fit(Xtrain, Ytrain, nb_epoch=nb)
# print model.evaluate(Xtest, Ytest)