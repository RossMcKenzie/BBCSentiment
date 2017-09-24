import pickle
import numpy as np
from keras.preprocessing import sequence
from random import shuffle

def genData(filePathX, filePathY, maxlen = 200, minValue = 1, maxValue = 20000):
    with open (filePathX, 'rb') as fp:
        X_full = pickle.load(fp)

    with open (filePathY, 'rb') as fp:
        Y_full = pickle.load(fp)

    #X_full = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8], [9,9], [10,10]]
    #Y_full = [1,2,3,4,5,6,7,8,9,10]
    #Create test and train sets
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
    X_train[X_train < minValue] = 1
    X_train[X_train > maxValue] = 1
    X_test[X_test < minValue] = 1
    X_test[X_test > maxValue] = 1
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)


    return X_train, Y_train, X_test, Y_test

def makePredictArray(var, support, size=100):
    np.random.shuffle(support)
    support = support[:size]
    X_pred = [np.tile(var,(size,1)), support]
    return X_pred

def getSimilarity(array, model):
    outData = model.predict(array)
    ans = np.mean(outData)
    return ans

def testAccuracy(X_val, Y_val, X_train, Y_train, model, size):
    outs = np.zeros(len(Y_val))
    posSupport = X_train[Y_train==1]
    negSupport = X_train[Y_train==0]
    for i in range(len(X_val)):
        posArr = makePredictArray(X_val[i], posSupport, size)
        negArr = makePredictArray(X_val[i], negSupport, size)
        posPred = getSimilarity(posArr, model)
        negPred = getSimilarity(negArr, model)
        outs[i] = (posPred-negPred)>0
    return np.sum(Y_val == outs)/len(Y_val)
