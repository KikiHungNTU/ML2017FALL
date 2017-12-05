# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 02:26:57 2017

@author: Ouch
"""
import numpy as np
import pandas as pd

print('Loading Data...')
def get_test_data(file):
    with open(file, encoding="utf-8") as f:
        f.readline() # header
        return [s[s.index(',')+1:] for s in f.readlines()]

def get_train_data(label_file, no_label_file):
    labels, label_text, unlabel_text = [], [], []
    with open(label_file, encoding="utf-8") as f:
        for s in f.readlines():
            label, text = s.split('+++$+++')
            labels.append(int(label))
            label_text.append(text)
    
    with open(no_label_file,encoding="utf-8") as f:
        unlabel_text = f.readlines()
    return labels, label_text, unlabel_text

path = 'C:/Users/chich/Desktop/ML2017FALL/4/data/'
test_file = path + 'testing_data.txt'
label_file = path + 'training_label.txt'
no_label_file = path + 'training_nolabel.txt'

test_text = get_test_data(test_file)
labels, label_text, unlabel_text = get_train_data(label_file, no_label_file)

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

print('Making WordTable...')
#把所有資料拿來做文字對照表
Text = label_text +test_text #+unlabel_text
WordTable = split_word(Text)
WordTableUni = np.unique(WordTable)
#轉換
word2idx = {}
for idx, word in enumerate(WordTableUni):
    word2idx[word] = idx+1

#存WordTable
WordFile = open(path+'WordTable.txt','w', encoding = 'utf8') 
for row in range(len(WordTable)):
    WordFile.write(WordTable[row] + '\n')
WordFile.close()

##讀Word2idx
#wTable = []
#idx = 0
#with open(path+'WordTable.txt', 'r', encoding = 'utf8') as wFile:
#    for i in wFile.readlines() :
#        i = i.strip('\n')
#        wTable.append([])
#        wTable[idx] = i
#        idx = idx+1
#wTableUni = np.unique(wTable)
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

#轉換train,test to idx
text_train = text_to_idx(label_text,word2idx)
text_test = text_to_idx(test_text,word2idx)

print('Training...')

from keras import regularizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Dropout
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

word2vec_dim = 300
maxlen = 37
text_train = sequence.pad_sequences(text_train, maxlen=maxlen)
text_test = sequence.pad_sequences(text_test, maxlen=maxlen)
max_features = len(word2idx)

model = Sequential()
#model.add(Embedding(max_features, 128))
#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.add(Embedding(max_features, output_dim = word2vec_dim, embeddings_initializer = 'RandomNormal', input_length=None))
model.add(LSTM(word2vec_dim ,dropout = 0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation = 'sigmoid',activity_regularizer=regularizers.l2(0.001)))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#標一下最好ㄉ
checkpoint = ModelCheckpoint(path+'models/model_acc{val_acc:.4f}_epoch{epoch:03d}.hdf5', monitor='val_acc', verbose=0,save_best_only=False, save_weights_only=False, mode='max', period=1)
train_history = model.fit(text_train ,labels ,batch_size = 256, epochs = 3,shuffle='True',validation_split=0.1,callbacks=[checkpoint] )
print('Predicting...')
y_test = model.predict_classes(text_test)
print('Writing...')
df = pd.DataFrame({'label':y_test.T[0], 'id': range(len(y_test))}).to_csv(path+'Predict.csv', index = False)
