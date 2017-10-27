# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:10:55 2017

@author: Ouch
"""

import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import sys

Train_original = sys.argv[1]
Test_original = sys.argv[2]

X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test = sys.argv[5]

#讀data set
x_train = pd.read_csv('X_train').as_matrix()
y_train = pd.read_csv('Y_train').as_matrix()
x_test = pd.read_csv('X_test').as_matrix()

#xgBoost
xgbModel = XGBClassifier(max_depth = 6, nthread = 4, learning_rate = 0.1, n_estimators = 1000, gamma = 1, objective= 'binary:logistic' ).fit(x_train,y_train)
#modelfit = xgb.train(xgbModel, xgbtrain,num_rounds, evals)
y_test = xgbModel.predict(x_test)
#Feature_Importance = xgbModel.feature_importances_
#print(Feature_Importance)

outputFile = str(sys.argv[6])
df = pd.DataFrame({'label': y_test, 'id': np.arange(y_test.shape[0])+1}).to_csv(outputFile, index=False)

#df = pd.DataFrame({'label': y_test, 'id': np.arange(y_test.shape[0])+1}).to_csv('outputF.csv', index=False)
