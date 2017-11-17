# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D,ZeroPadding2D, Flatten, Input, Reshape, BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.datasets import mnist
from keras import regularizers
from keras.layers.advanced_activations import PReLU, LeakyReLU
import time
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import sys

x_train = sys.argv[1]
#x_test = sys.argv[2]
#outputFile = str(sys.argv[3])

#ACCearlyStopping=EarlyStopping( monitor = 'val_acc',patience = 2)
#LOSSearlyStopping=EarlyStopping( monitor = 'val_loss',patience = 2)
#def data_augmentation(X,Y):
#    x_train_Augmentation = np.fliplr(X)
#    y_train_Augmentation = Y.copy()
#    x_all = np.concatenate((x_train, x_train_Augmentation), axis=0)
#    Y_all = np.concatenate((y_train_Augmentation, y_train_Augmentation), axis=0)
#    return(x_all,Y_all)

#data set
data_train = pd.read_csv(x_train)
#x_test = pd.read_csv(x_test)
y_train = data_train.iloc[:,0]
#print(y_train)
def TransInt(s):
    return np.reshape([int(tok) for tok in s.split()], (48, 48))

data_train.feature = data_train.feature.apply(TransInt) / 255
#x_test.feature = x_test.feature.apply(TransInt) / 255
x_train = np.stack(data_train.feature.as_matrix()).reshape((-1, 48, 48, 1))
#x_test = np.stack(x_test.feature.as_matrix()).reshape((-1, 48, 48, 1))
#y_valid = y_train.iloc[0:4156]
#data_augmentation(x_train,y_train)

#print(np.shape(data_train), np.shape(x_test))

#x_train = data_train[0:,1]
#x_train = x_train.reshape(x_train.shape[0], 48*48)
#print(x_train[0,0])

y_train = np_utils.to_categorical(y_train,7)

x_valid = x_train[0:2078,:,:,:]
y_valid = y_train[0:2078,:]
##
x_train = x_train[2078:,:,:,:]
y_train = y_train[2078:,:]

model = Sequential()
model.add(Conv2D(64, (5, 5), padding='same',input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=1./20))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add( ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=1./20))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=1./20))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

#model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=1./20))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=1./20))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu' ))
#model.add(Dropout(0.5))

model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001) ))
model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001) ))
model.add(Dropout(0.5))

model.add(Dense(7))
model.add(Activation('softmax'))
# opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(lr=0.001) #好
#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)   #快
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

data = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=30,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

train_history = model.fit_generator(data.flow(x_train, y_train, 128),steps_per_epoch=len(x_train)// 128,validation_data=(x_valid, y_valid), epochs=500 )
#model.fit(x_train,y_train, epochs = 200, batch_size = 256 ,shuffle='True', validation_split = 0.1, class_weight='auto' )
#y_test = model.predict_classes(x_test)
#print(y_test.T)

#model.summary()

##畫圖
#show_train_history(train_history, 'acc', 'val_acc')  
#show_train_history(train_history, 'loss', 'val_loss') 

###畫出Confusion Matrix
#print("\t[Info] Display Confusion Matrix:")  
#print(pd.crosstab(y_valid, y_test, rownames=['label'], colnames=['predict'],margins = True)) 


#df = pd.DataFrame({'label':y_test.T, 'id': np.arange(y_test.shape[0])}).to_csv('Predict_Batch.csv', index = False)

# save model
model_json = model.to_json()
# save structure
open('model_architecture.json', 'w').write(model_json)
# save weight
model.save_weights('model_weights.h5')
