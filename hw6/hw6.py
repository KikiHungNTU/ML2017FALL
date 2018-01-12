# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:14:18 2018

@author: Ouch
"""

import numpy as np
import pandas as pd
import csv
import sys

#path = 'C:/Users/chich/Desktop/ML2017FALL/6/'
#test = pd.read_csv(path + 'test_case.csv')
#test_1 = test['image1_index']
#test_2 = test['image2_index']
#image = np.load('image.npy')

imageFile = sys.argv[1]
testFile = sys.argv[2]
outputFile = sys.argv[3]

test = pd.read_csv(testFile)
test_1 = test['image1_index']
test_2 = test['image2_index']
image = np.load(imageFile)

imageList = image.astype('float32') / 255.
imageList = imageList.reshape((len(imageList), -1))
#pro = np.prod(imageList.shape[1:])
#imageList = imageList.reshape((len(imageList),pro))

print (imageList.shape)

import keras
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from sklearn import cluster
from keras.models import load_model

encoding_dim = 128

input_img = Input(shape=(784,))

encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)

encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)

decoded = Dense(784, activation='tanh')(decoded)
#decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
#
#autoencoder.summary()
#encoder.summary()
autoencoder.compile(optimizer='adam', loss='mse')
#autoencoder.fit(image, image, epochs=100, batch_size=512, shuffle=True, validation_split=0.1 )
#autoencoder.save('autoencoder.h5')

autoencoder.load_weights('autoencoder8.h5')
encoded__ = encoder.predict(imageList)

#Cluster-Kmeans
clf = cluster.KMeans(init='k-means++', n_clusters=2)
clf.fit(encoded__)
clusters = clf.predict(encoded__)

ans = []
for i in range(len(test)):
    if clusters[test_1[i]]==clusters[test_2[i]]:
        ans.append(1)
    else:
        ans.append(0)       

with open(outputFile, 'w+') as f:
    row = csv.writer(f, delimiter=',', lineterminator='\n')
    row.writerow(["ID","Ans"])
    n = len(ans)
    for i in range(n):
        row.writerow([i,ans[i]])
f.close()

#visualization = np.load('visualization.npy')
#visual_encoded__ = encoder.predict(visualization)
#
#clf = cluster.KMeans(init='k-means++', n_clusters=2)
#clf.fit(visual_encoded__ )
#clusters = clf.predict(visual_encoded__ )
#import matplotlib.pyplot as plt
#for i in range(clusters.shape[0]):
#    if clusters[i] == 1:
#        plt.scatter(visual_encoded__[i,56],visual_encoded__[i,82],c = 'green',s = 6)
#    else:
#        plt.scatter(visual_encoded__[i,56],visual_encoded__[i,82],c = 'blue',s = 6)
#        
#plt.show()
#
#for i in range(clusters.shape[0]):
#    if i > 4999:
#        plt.scatter(visual_encoded__[i,56],visual_encoded__[i,82],c = 'green',s = 6)
#    else:
#        plt.scatter(visual_encoded__[i,56],visual_encoded__[i,82],c = 'blue',s = 6)
