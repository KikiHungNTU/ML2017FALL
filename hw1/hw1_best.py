# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 00:48:38 2017

@author: Ouch
"""

###Private Best Pick CO、NO2、O3、PM10、PM2.5、SO2、SquareTerm###
import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import pandas as pd

#Train
data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時471 #連續5小時475
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

#scale
#scale = []
#for i in range(1,len(x[0])):
#    scale = x[:,i]
#    var = np.var(scale)**0.5
#    mean = np.mean(scale)
#    x[:,i] = (x[:,i] - mean ) / var

##
X = [1] * len(x)
SquareTerm = [1]*len(x)

for row in range(len(x)):
    X[row] = []
    SquareTerm[row] = []
#
##print(np.shape(X))
for row in range(len(x)):
    #Lasso
#     X[row] = x[row][np.r_[18:27,45:54,54:63,63:72,72:81,81:90,108:117]]
#     SquareTerm[row] = x[row][np.r_[72:81,99:108,117:126]]
#     #選CO+NMHC+NO+NO2+O3+PM10+PM2.5+SO2
#     X[row] = x[row][np.r_[18:27,27:36,45:54,63:72,72:81,81:90,108:117]] #5652,63
#     選PM2.5、PM10
#     X[row] = x[row][np.r_[72:81,81:90]]
#     SquareTerm[row] = x[row][np.r_[72:81,81:90]]
     #選CO+NO2+O3+PM10+PM2.5+SO2
     X[row] = x[row][np.r_[18:27,45:54,63:72,72:81,81:90,108:117]] 
     SquareTerm[row] = x[row][np.r_[18:27,45:54,63:72,72:81,81:90,108:117]]

x = np.array(X)
SquareTerm = np.array(SquareTerm)
#
##print(SquareTerm)
##print(x.shape)
##print(np.shape(PM252))
##print(x)
#
#
## add square term
x = np.concatenate((x,SquareTerm**2) , axis=1) #5652選Feature#

#print(x.shape)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1) #5652,127
#print(x.shape)

#filename = "Data_Train.csv"
#text = open(filename, "w+")
#s = csv.writer(text,delimiter=',',lineterminator='\n')
#
#for i in range(len(x)):
#    s.writerow(x[i]) 
#text.close()
#filename = "y_Train.csv"
#text = open(filename, "w+")
#s = csv.writer(text,delimiter=',',lineterminator='\n')
#s.writerow(y) 
#text.close()

w = np.zeros(len(x[0]))
l_rate = 10000
repeat = 100000
Lambda = 1

# use close form to check whether ur gradient descent is good
# however, this cannot be used in hw1.sh 
# w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    #預測y
    hypo = np.dot(x,w)
    loss = hypo - y  #預測-實際
        
    #RMSE
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    
    #Adagrad+Regularized
    gra = np.dot(loss,x)+2*Lambda*np.sum(w)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
       
    w = w - l_rate * gra/ada # update weight;decent
    print ('iteration: %d | Cost: %f  ' % ( i, cost_a ))
        
    # save model
np.save('hw1_best.npy',w)
# read model
w = np.load('hw1_best.npy')

#讀Test
inputFile = str(sys.argv[1])
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
#
##print(np.shape(test_X))
#
for row in range(len(test_x)):
    test_X[row] = test_x[row][np.r_[18:27,45:54,63:72,72:81,81:90,108:117]]

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

outputFile = sys.argv[2]
#filename = "predict.csv"
text = open(outputFile, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
