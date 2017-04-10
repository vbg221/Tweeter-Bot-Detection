# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:30:35 2017

This program uses Random Forest Algorithm to predict whether or not a twitter account is bot.
It reads data from a filtereed CSV file.

@author: vbg22
"""


import numpy as np
#from numpy import genfromtxt
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
'''
f = open('C:/Users/vbg22/Downloads/Machine Learning Project/FinalData_filtered.csv','rb')
fo = open('C:/Users/vbg22/Downloads/Machine Learning Project/FinalData_filter.csv','wb')


# go through each line of the file
for line in f:
    bits = line.split(',')
    # change second column
    if bits[8] == "TRUE":
        bits[8] = '1'
    else:
        bits[8] = '0'
    # join it back together and write it out
    fo.write(','.join(bits))
f.close()
fo.close()
'''


data = pd.read_csv('C:/Users/vbg22/Downloads/Machine Learning Project/shity.csv') #, encoding = 'latin1'

                  
                  
data = data.replace(np.nan, '', regex = True)

np.any(np.isnan(data))
np.all(np.isfinite(data))

data.shape
data.dropna()

#np.isnan(data.any()) #and gets False
#np.isfinite(data.all()) #and gets True
            
#data.replace(to_replace = True, value = 1)
#data.replace(to_replace = False, value = 0)

type(data.ix[0]['screen_name'])
#data.astype(float)

#words=['','Null', 'NAN', 'None', 'none', 'null','nan'] #words to be searched in the description
#for data in range (0, len(data['url'])):
    #data.replace(to_replace = "nan", value = '0')
    #data.replace(to_replace = "null", value = '0')
#    data.replace(to_replace = words, value = 0)
#    data.replace(to_replace = re.match("http*", 1, 1), value  = 1, regex = True)



'''
#data['url']=[(((data.iloc[i]['url']=='"nan", "None", "null"')==False)*1) for i in range (len(data))]
words=['Null', 'NAN', 'None', 'none', 'null','nan'] #words to be searched in the description
pat = '|'.join(map(re.escape, words)) #mapping function
data = data.str.lower() #converting descritpion to lower case
data=data.description.str.contains(pat)*1 
'''

msk = np.random.rand(len(data)) < 0.8
                    
train = data[msk]

test = data[~msk]
#test = test.replace(np.nan, '', regex = True)

cols = ['screen_name','location','followers_count','friends_count','listedcount','favourites_count','verified','statuses_count','status','name']
colRes = ['bot']

trainArr = train.as_matrix(cols) #training array
#trainArr = trainArr.replace(np.nan, 0, regex = True)
trainRes = train.as_matrix(colRes) # training results
                         
rf = RandomForestClassifier(n_estimators=100) # initialize
rf.fit(trainArr, trainRes) # fit the data to the algorithm

      
      
testArr = test.as_matrix(cols)
#testArr = testArr.replace(np.nan, 0, regex = True)
results = rf.predict(testArr)

test['predictions'] = results
test.head()

#rf.score(test['bot'], test['predictions'])
rf.score(testArr, test['bot'])


def calculateAccuracy(bot, predictions):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range (0, len(test)):
        if bot[i] == predictions[i]:
            if bot[i] == True:
                tp = tp + 1
            elif bot[i] == False:
                tn = tn + 1
        elif bot[i] == True:
            fn = fn + 1
        elif bot[i] == False:
            fp = fp + 1
    
    return [tp, fp, tn, fn]

truePositive, falsePositive, trueNegative, falseNegative = calculateAccuracy(test['bot'], test['predictions'])
tpr = (truePositive) / (truePositive + falseNegative)
fpr = (falsePositive) / (falsePositive + trueNegative)
