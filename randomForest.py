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
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datacleaner import autoclean


#from sklearn.linear_model import LogisticRegression


data = pd.read_csv('shity.csv') #, encoding = 'latin1'
data2 = pd.read_csv('Final_test.csv')
data = data.dropna()
#data2.dropna() 
data = data.replace(np.nan, '', regex = True)
data2 = data2.replace(np.nan, '', regex = True)
"""
for i in range(0,len(data)):
    if(np.isnan(data['screen_name'][i])):
        data['screen_name'][i] = 0
    if(np.isnan(data['location'][i])):
"""
#np.any(np.isnan(data))
#pd.any(pd.isnull(data))
#np.all(np.isfinite(data))


#data.shape

#np.isnan(data.any()) #and gets False
#np.isfinite(data.all()) #and gets True
            
data.replace(to_replace = True, value = 1)
data.replace(to_replace = False, value = 0)
data2.replace(to_replace = True, value = 1)
data2.replace(to_replace = False, value = 0)
#data2.replace(to_replace = Nan, value = 0)
#data2.replace(to_replace = "", value = '')

#type(data['favourites_count'][1])


#type(data.ix[0]['screen_name'])
#data['screen_name'].astype(float)
data.astype(float)
data2.astype(float)

data.dropna()
data2.dropna()

'''
#data['url']=[(((data.iloc[i]['url']=='"nan", "None", "null"')==False)*1) for i in range (len(data))]
words=['Null', 'NAN', 'None', 'none', 'null','nan'] #words to be searched in the description
pat = '|'.join(map(re.escape, words)) #mapping function
data = data.str.lower() #converting descritpion to lower case
data=data.description.str.contains(pat)*1 
'''

#msk = np.random.rand(len(data)) < 0.8
         
                    
                     
train = data
test = data2

train.dropna()
test.dropna()
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
results = rf.predict(test)
#results.toColumn()

df = pd.read_csv('test_data_4_students.csv')
df2 = df['id']
df2 = pd.concat([df['id'],pd.DataFrame(results.T)], axis=1, join='inner')

df2.columns = ['Id','Bot']

df2.dropna()
df2['Id'].astype(int)
df2.to_csv('Output.csv', sep=',', index=False)
#df2['bot'] = rf.predict(testArr)
#np.shape(df2)
#test['bot'] = results
#test.head(10)

'''
#rf.score(test['bot'], test['predictions'])
rf.score(testArr, test['bot'])


def calculateAccuracy(bot, predictions, length):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range (0, length):
        if bot.iloc[i] == predictions.iloc[i]:
            if bot.iloc[i] == True:
                tp = tp + 1
            elif bot.iloc[i] == False:
                tn = tn + 1
        elif bot.iloc[i] == True:
            fn = fn + 1
        else: #bot[i] == False:
            fp = fp + 1
    
    return [tp, fp, tn, fn]

len(test)

truePositive, falsePositive, trueNegative, falseNegative = calculateAccuracy(test['bot'], test['predictions'], len(test))
tpr = (truePositive) / (truePositive + falseNegative)
fpr = (falsePositive) / (falsePositive + trueNegative)

tpr = [0.0, tpr, 1.0]
fpr = [0.0, fpr, 1.0]

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print(classification_report(test['bot'], test['predictions']))
'''
"""
# load dataset
dta = sm.datasets.fair.load_pandas().data

# add "affair" column: 1 represents having affairs, 0 represents not
dta['affair'] = (dta.affairs > 0).astype(int)
"""