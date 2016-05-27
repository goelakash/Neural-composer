#!/usr/bin/python

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
step = 3
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

starters = []
for i in range(0,5):
	starters.append(X[i*1000])

print X[0].shape
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

nb = 5
model.fit(Xtrain, Ytrain, nb_epoch=nb)
print model.evaluate(Xtest, Ytest)

gen_songs = []

for st in starters:
    res = st              
    pos = 0
    for i in range(0,10000):
        pred = model.predict(np.asarray([res[pos:pos+maxlen]]))
        for i in range(0,len(pred[0])):
        	pred[0][i] = int(pred[0][i])
        	if pred[0][i]<40:
        		pred[0][i]=0

        res = np.vstack((res,pred))
        pos += 1

    #result is a state matrix, so convert
    print res.shape
    new_song = matrix_to_midi(res)
    gen_songs.append(copy(new_song))

for i in range(0,len(gen_songs)):
	midi.write_midifile("gensong_"+str(i)+".mid",gen_songs[i])