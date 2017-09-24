'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
from Funcs import genData

''' Basic LSTM network to determine BBC news article sentiment.
Based https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py'''
np.random.seed(1337)
max_features = 20000
maxlen = 200
batch_size = 442
epochs = 20
width = 300
dropout = 0.3
for i in range(1):
    #Open data
    X_train, Y_train, X_test, Y_test = genData('Data/BBCSequencesX.pkl', 'Data/BBCSequencesY.pkl')

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, maxlen, dropout=dropout))
    model.add(LSTM(width, dropout_W=0.2, dropout_U=dropout, return_sequences=True))
    model.add(LSTM(width, dropout_W=0.2, dropout_U=dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])

    print('Train...')
    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, validation_data=(X_test, Y_test))
    model.save("BBCLSTM040417.mdl")
    score, acc = model.evaluate(X_test, Y_test)
    print('Test score:', score)
    print('Test accuracy:', acc)

    plt.figure(i)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
plt.show()
#loss = model.evaluate(X_test, Y_test)
#print('Test loss:', loss)
