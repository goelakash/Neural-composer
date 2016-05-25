from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop
import time
import numpy as np

from format import *

SONG_NAME = "bebop.midi"
song_matrix = np.asarray(midi_to_matrix(SONG_NAME))

maxlen = 50
states = np.asarray([[x[0] for x in row] for row in song_matrix])

sentences = []
next_chars = []
step = 5
for i in range(0, song_matrix.shape[0] - maxlen, step):
    sentences.append(states[i: i + maxlen])
    next_chars.append(states[i + maxlen])
print('nb sequences:', len(sentences))


X = np.zeros((len(sentences), maxlen, song_matrix.shape[1]))
Y = np.zeros((len(sentences), song_matrix.shape[1]))

# sentence = collection of time slices
# char = time slice
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char
    Y[i] = next_chars[i]

n = int(len(X)*0.9)

Xtrain = X[:n]
Ytrain = Y[:n]
Xtest = X[n:]
Ytest = Y[n:]

print Xtrain.shape
print Ytrain.shape

# print Xtrain[:2]
# print Ytrain[:2]

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(maxlen, 128)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation('relu'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

print "Compiling"
start = time.time()
model.compile(loss='mean_squared_error', optimizer=sgd)
elapsed = time.time() - start
print elapsed/60, ' and ', elapsed%60

nb = 10
model.fit(Xtrain, Ytrain, nb_epoch=nb)
print model.evaluate(Xtest, Ytest)