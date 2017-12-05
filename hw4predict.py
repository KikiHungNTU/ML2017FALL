# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:21:07 2017

@author: Ouch
"""
from keras.preprocessing import sequence
from keras.models import model_from_json, load_model
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
import sys

#取得test_text
def get_test_data(file):
    with open(file, encoding="utf-8") as f:
        f.readline() # header
        return [s[s.index(',')+1:] for s in f.readlines()]
path = 'C:/Users/chich/Desktop/ML2017FALL/4/data/'
test_file = path + 'testing_data.txt'

test_text = get_test_data(test_file)

#分割文字
def split_word(text):
    WordTable = []
    a = len(text)
    for row in range(a):
        Word = text[row].split(' ')
        count = len(Word)    
        for i in range(count):
            WordTable.append(Word[i].strip())
    return WordTable

#讀WordTable
wTable = []
idx = 0
with open(path+'WordTable.txt', 'r', encoding = 'utf8') as wFile:
    for i in wFile.readlines() :
        i = i.strip('\n')
        wTable.append([])
        wTable[idx] = i
        idx = idx+1
wTableUni = np.unique(wTable)     

#轉換
word2idx = {}
for idx, word in enumerate(wTableUni):
    word2idx[word] = idx+1

#把text轉成idx
def text_to_idx(text,word2idx):
    text_ = []
    text_idx = []
    for row in range(len(text)):
        text_idx.append([])
        Word = text[row].split(' ')
        text_.append(Word)
        for token in text_[row]:
            idx = word2idx[token.strip()]
            text_idx[row].append(idx)
    return text_idx
text_test = text_to_idx(test_text,word2idx)


word2vec_dim = 300
maxlen = 37
text_test = sequence.pad_sequences(text_test, maxlen=maxlen)
max_features = len(word2idx)

model = Sequential()
model = load_model(path+'models/'+'model.h5')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
y_test = model.predict_classes(text_test)
df = pd.DataFrame({'label':y_test.T[0], 'id': range(len(y_test))}).to_csv(path+'Predict.csv', index = False)
