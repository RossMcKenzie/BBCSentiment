import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, Input, merge
from keras.layers import LSTM
from keras import backend as K
import matplotlib.pyplot as plt
from Funcs import *
from random import randint

'''Uses a siamese LSTM network to tell news articles with different sentiment appart
Tested due to lack of data for training standard network.
Based on code and info from https://sorenbouma.github.io/blog/oneshot/'''

max_features = 20000
maxlen = 200
batch_size = 500
epochs = 20
width = 100
dropout = 0.3
input_shape = (200)
left_input = Input([maxlen])
right_input = Input([maxlen])
X_trainSet, Y_trainSet, X_testSet, Y_testSet = genData('Data/BBCSequencesX.pkl', 'Data/BBCSequencesY.pkl')

X_trainSet = X_trainSet[:50]
X_testSet = X_testSet[:50]

with open('testX110417.pkl', 'wb') as pickle_file:
    pickle.dump(X_testSet, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
with open('testY110417.pkl', 'wb') as pickle_file:
    pickle.dump(Y_testSet, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

X_train = [np.zeros((len(X_trainSet)**2, maxlen)) for i in range(2)]
Y_train = np.zeros(len(X_trainSet)**2)
X_test = [np.zeros((len(X_testSet)**2, maxlen)) for i in range(2)]
Y_test = np.zeros(len(X_testSet)**2)

for i in range(len(X_trainSet)):#make paired data
    for j in range(len(X_trainSet)):
            index = ((i+1)*(j+1))-1
            X_train[0][index] = X_trainSet[i]
            X_train[1][index] = X_trainSet[j]
            Y_train[index] = Y_trainSet[i] == Y_trainSet[j]


for i in range(len(X_testSet)):#make paired data
    for j in range(len(X_testSet)):
            index = ((i+1)*(j+1))-1
            X_test[0][index] = X_testSet[i]
            X_test[1][index] = X_testSet[j]
            Y_test[index] = Y_testSet[i] == Y_testSet[j]

#Build one half of LSTM
model = Sequential()
model.add(Embedding(max_features, width, dropout=dropout))
model.add(LSTM(width, dropout_W=0.2, dropout_U=dropout, return_sequences=True))  # try using a GRU instead, for fun
model.add(LSTM(width, dropout_W=0.2, dropout_U=dropout))
model.add(Dense(width,activation="sigmoid"))

#encode each of the two inputs into a vector with the convnet
encoded_l = model(left_input)
encoded_r = model(right_input)

#merge two encoded inputs with the l1 distance between them
L1_distance = lambda x: K.abs(x[0]-x[1])
both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1,activation='sigmoid')(both)
siamese_net = Model(input=[left_input,right_input],output=prediction)
#optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)

siamese_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
history = siamese_net.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, validation_data=(X_test, Y_test))
siamese_net.save("BBCSiamLSTM110417.mdl")
score, acc = siamese_net.evaluate(X_test, Y_test)
print('Val score:', score)
print('Val accuracy:', acc)
out = testAccuracy(X_testSet, Y_testSet, X_trainSet, Y_trainSet, siamese_net, 10)
print('Test accuracy:', out)

plt.figure(i)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
