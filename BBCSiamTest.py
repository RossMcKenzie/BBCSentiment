from Funcs import *
from keras.models import Model, load_model
siamese_net = load_model("BBCSiamLSTM100417.mdl")
X_trainSet, Y_trainSet, X_testSet, Y_testSet = genData('Data/BBCSequencesX.pkl', 'Data/BBCSequencesY.pkl')
out = testAccuracy(X_testSet, Y_testSet, X_trainSet, Y_trainSet, siamese_net, 10)
print('Test accuracy:', out)
