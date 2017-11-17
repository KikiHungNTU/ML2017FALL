# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:05:38 2017

@author: Ouch
"""

from keras.models import model_from_json, load_model
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
import sys

x_test = sys.argv[1]
outputFile = str(sys.argv[2])

#path = 'C:/Users/chich/Desktop/ML2017FALL/3/Final/'
#data_train = pd.read_csv(path + 'train.csv')
x_test = pd.read_csv(x_test)
#y_train = data_train.iloc[:,0]
#print(y_train)
def TransInt(s):
    return np.reshape([int(tok) for tok in s.split()], (48, 48))

#data_train.feature = data_train.feature.apply(TransInt) / 255
x_test.feature = x_test.feature.apply(TransInt) / 255
#x_train = np.stack(data_train.feature.as_matrix()).reshape((-1,48,48,1))
x_test = np.stack(x_test.feature.as_matrix()).reshape((-1, 48,48,1))
#y_valid = y_train.iloc[0:2078]
#y_train = np_utils.to_categorical(y_train,7)
#x_valid = x_train[0:2078,:,:,:]
#y_valid = y_train[0:2078,:]
#
#x_train = x_train[2078:,:,:,:]
#y_train = y_train[2078:,:]
opt = Adam(lr=0.001) #好

model = Sequential()
model = model_from_json(open('model_architecture.json').read())
model.load_weights('model_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

y_test = model.predict_classes(x_test)

df = pd.DataFrame({'label':y_test.T, 'id': np.arange(y_test.shape[0])}).to_csv(outputFile, index = False)
