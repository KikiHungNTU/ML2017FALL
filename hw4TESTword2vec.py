# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 02:26:57 2017

@author: Ouch
"""
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer,text_to_word_sequence

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
#
##分割文字
def split_word(text):
    WordTable = []
    a = len(text)
    for row in range(a):
        Word = text[row].split(' ')
        count = len(Word)    
        for i in range(count):
            WordTable.append(Word[i].strip())
    return WordTable

#把句子拿來做Word2vec
all_sentences = label_text +test_text +unlabel_text
text_train_seq = []
text_test_seq = []
for i in range(len(label_text)):
    text_train_seq.append(label_text[i].split())
    #text_train_seq.append(text_to_word_sequence(label_text[i],filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'+"'",split=" "))

for i in range(len(test_text)):
    text_test_seq.append(test_text[i].split())
    #text_test_seq.append(text_to_word_sequence(test_text[i],filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'+"'",split=" "))

##存句子給word2vec
#SentenceFile = open(path+'all_sentences','w', encoding = 'utf8') 
#for row in range(len(all_sentences)):
#    SentenceFile.write(all_sentences[row])
#SentenceFile.close()
#
##讀讀看
#sentenceTable = []
#with open(path+'all_sentences', 'r', encoding = 'utf8') as wFile:
#    for i in wFile.readlines() :
#        i = i.strip('\n')
#        sentenceTable.append([i])

print('Processing--Word2vec--')
from gensim.models import word2vec
import logging
word_dim = 300
max_len_seq = 40

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#sentences = word2vec.Text8Corpus(all_sentences)
#word_model = word2vec.Word2Vec([s.split() for s in all_sentences], size=word_dim,min_count=3, iter=9)
#word_model.save(path+'models/word2vec')
#print('word2vec DONE')

# #Load Word2vec


word_model = word2vec.Word2Vec.load(path+'models/word2vec')
word2index = {word: ind + 1 for ind, word in enumerate(word_model.wv.index2word)} # 0 for padding
word_vectors = [np.zeros(word_dim)]
for word in word_model.wv.index2word:
    word_vectors.append(word_model[word])
word_vectors = np.stack(word_vectors)
text_train = [[word2index.get(s, 0) for s in line] for line in text_train_seq]
text_test = [[word2index.get(s, 0) for s in line] for line in text_test_seq]
#補0
#text_train = np.array([np.concatenate([np.array([0.0]*word_dim*(max_len_seq-row.shape[0])).reshape(max_len_seq-row.shape[0],word_dim),row]) for row in text_train])
#text_test = np.array([np.concatenate([np.array([0.0]*word_dim*(max_len_seq-row.shape[0])).reshape(max_len_seq-row.shape[0],word_dim),row]) for row in text_test])
print('word_vectors.shape', word_vectors.shape)

print('Training...')
import keras
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Dropout
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

max_len_seq = 40
text_train = pad_sequences(text_train, maxlen=max_len_seq)
text_test = pad_sequences(text_test, maxlen=max_len_seq)

def get_model(word_vectors):
    model = Sequential()
    model.add(Embedding(word_vectors.shape[0], word_vectors.shape[1], trainable=False,
                            embeddings_initializer=keras.initializers.Constant(word_vectors))) # Using pretained word embedding
    model.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

#標一下最好ㄉ
model = get_model(word_vectors)
checkpoint = ModelCheckpoint(path+'models/model_acc{val_acc:.4f}_epoch{epoch:03d}.hdf5', monitor='val_acc', verbose=0,save_best_only=False, save_weights_only=False, mode='max', period=1)
#train_history = model.fit(text_train ,labels ,batch_size = 1024, epochs = 15,shuffle='True',validation_split=0.1,callbacks=[checkpoint] )
model.load_weights('models/model_acc0.8235_epoch004.hdf5')
print('Predicting...')
y_test = model.predict_classes(text_test, batch_size=256, verbose=1)
print('Writing...')
df = pd.DataFrame({'label':y_test.T[0], 'id': range(len(y_test))}).to_csv(path+'Predict.csv', index = False)

