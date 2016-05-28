#!/usr/bin/python

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop
import time
import numpy as np
import os
from format import *

song_list = []
for r,sd,fl in os.walk("music"):
	song_list = fl

sentences = []
next_chars = []
song_matrix_list = []

for SONG_NAME in song_list:
	print "music/"+SONG_NAME
	song_matrix_list.extend(midi_to_matrix("music/"+SONG_NAME))

for song_matrix in song_matrix_list:
	maxlen = 50
	states = song_matrix

	step = 10
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
for i in range(0,3):
	starters.append(X[i*10000])

print X[0].shape

# model = Sequential()
# model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, 128)))
# model.add(Dropout(0.2))
# model.add(LSTM(512, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(256))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Dense(128))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# print "Compiling"
# start = time.time()
# model.compile(loss='mean_squared_error', optimizer=sgd)
# elapsed = time.time() - start
# print int(elapsed/60), 'minutes and ', elapsed%60, " seconds"

import cPickle as pickle
model = pickle.load(open("untraimed_model_512_512_256_128__nb_eq_10.p","rb"))

nb = 100
model.fit(Xtrain, Ytrain, nb_epoch=nb)
print model.evaluate(Xtest, Ytest)

import sys
sys.setrecursionlimit(10000)
pickle.dump(model,open("trained_model.p","wb"),protocol=pickle.HIGHEST_PROTOCOL)

results = []
gen_songs = []


for st in starters:
    res = st
    pos = 0
    for i in range(0,50000):
        pred = model.predict(np.asarray([res[pos:pos+maxlen]]))
        res = np.vstack((res,pred))
        pos += 1
    print res.shape
    results.append(copy(res))

for i in range(0,len(results)):
    new_song = matrix_to_midi(results[i].astype('int64',copy=False))
    print new_song
    midi.write_midifile("gensong_"+str(i)+".mid",new_song)
