from __future__ import print_function
import numpy as np
import pickle
import random
from random import shuffle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout
from keras.layers import GRU

'''GRU NN for sentiment analysis.
Based on https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py'''

np.random.seed(1337)
max_features = 20000
maxlen = 200
batch_size = 221
epochs = 8
width = 300
dropout = 0.2

#Open data
with open ('Data/BBCSequencesX.pkl', 'rb') as fp:
    X_full = pickle.load(fp)

with open ('Data/BBCSequencesY.pkl', 'rb') as fp:
    Y_full = pickle.load(fp)

full = list(zip(X_full, Y_full))
shuffle(full)
train = full[:int(len(full)*0.9)]
test = [x for x in full if x not in train]
X_train, Y_train = zip(*train)
X_test, Y_test = zip(*test)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

#Create numpy arrays
print('Pad sequences')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, width, dropout=dropout))
model.add(GRU(width, dropout_W=0.2, dropout_U=dropout, return_sequences=True))  # try using a GRU instead, for fun
model.add(GRU(width, dropout_W=0.2, dropout_U=dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print('Train...')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
          validation_data=(X_test, Y_test))
model.save("BBCGRU040417.mdl")
score, acc = model.evaluate(X_test, Y_test)
print('Test score:', score)
print('Test accuracy:', acc)
