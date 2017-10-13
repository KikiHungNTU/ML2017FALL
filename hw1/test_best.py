# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 00:48:38 2017

@author: Ouch
"""

import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import pandas as pd


w = np.load(str(sys.argv[1]))

#讀Test
inputFile = str(sys.argv[2])
test_x = []
n_row = 0
text = open(inputFile ,"r")
#text = open('test.csv' ,"r")

row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        #9hr-2-11;5hr-2-7
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        #if n_row %18 in [2,3,5,7,8,9]:
            for i in range(2,11):
                if r[i] !="NR":
                    test_x[n_row//18].append(float(r[i]))
                else:
                    test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

#scale
#test_scale = []
#for i in range(1,len(test_x[0])):
#    test_scale = test_x[:,i]
#    test_var = np.var(scale)**0.5
#    test_mean = np.mean(scale)
#    test_x[:,i] = (test_x[:,i] - test_mean ) / test_var

#print(test_x)
#print(test_x.shape)

##選Feature
##print(np.shape(test_x))
test_X = [1] * len(test_x)
test_SquareTerm = [1]*len(test_x)

for row in range(len(test_x)):
    test_X[row] = []
#    test_SquareTerm[row] = []
#
##print(np.shape(test_X))
#
for row in range(len(test_x)):
    test_X[row] = test_x[row][np.r_[18:27,45:54,63:72,72:81,81:90,108:117]]
#
#
test_x = np.array(test_X)

##print(test_x)
#
## add square term of PM2.5
test_x = np.concatenate((test_x,test_x**2) , axis=1) #5652

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

outputFile = sys.argv[3]
#filename = "predict.csv"
text = open(outputFile, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
