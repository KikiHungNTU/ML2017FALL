# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
word = []

inputFile = str(sys.argv[1])
with open(inputFile, 'r') as file:
    for line in file:
        word = line.split()

wordsn = []
for i in word:
    if i not in wordsn:
        wordsn.append(i)
        
count = []
for i in wordsn:
    count.append(word.count(i))
    
wordList = []
for j in range(300):
    wordList.append(wordsn[j] + ' ' + str(j) + ' ' + str(count[j]))

#新建檔案寫入
thisFile = open('Q1.txt', 'w')
for item in wordList:
    thisFile.write(item)
    if wordList.index(item) < 299:
        thisFile.write('\n')
thisFile.close();