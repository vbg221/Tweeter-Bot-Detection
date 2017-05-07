
# coding: utf-8

# In[1]:


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

data = pd.read_csv('C:/Users/vbg22/Downloads/Pictures/binary and continouous.csv') #, encoding = 'latin1'
#data = pd.read_csv('C:/Users/vbg22/Downloads/Pictures/binary and continouous.csv', encoding = 'latin1') #, encoding = 'latin1'
data2 = pd.read_csv('C:/Users/vbg22/Downloads/Machine learning Project/Tweeter-Bot-Detection/test2.csv', encoding = 'latin1')
data = data.dropna()
data2.dropna()
data = data.replace(np.nan, '', regex = True)
data2 = data2.replace(np.nan, '', regex = True)
data


# In[2]:

data.shape


# In[3]:

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
            
#data.replace(to_replace = True, value = 1)
#data.replace(to_replace = False, value = 0)
#data2.replace(to_replace = True, value = 1)
#data2.replace(to_replace = False, value = 0)
#data2.replace(to_replace = Nan, value = 0)
#data2.replace(to_replace = "", value = '')

#type(data['favourites_count'][1])


#type(data.ix[0]['screen_name'])
#data['screen_name'].astype(float)
#data['name'].astype(float)
#data2['screen_name'].astype(float)
#data.astype(float)
#data2.astype(float)

#data = autoclean(data)
#data2 = autoclean(data2)

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
         

#,'url','created_at','lang','status','favourites_count','friends_count'
cols = ['screen_name','location','description','followers_count','listedcount']
colRes = ['bot']                    
                     
train = data
test = data2

train = train[cols]
test = test[cols]
test


# In[4]:

#test = test.replace(np.nan, '', regex = True)
#data = autoclean(data)
#test = autoclean(test)
test


# In[ ]:




# In[5]:

for header_name in cols:
    temp = list()
    for i in range(len(train)):
        temp.append(train.iloc[i][header_name].astype(float))
    train[header_name]= temp
print(train.shape)
train


# bigtemp = list()
# for header_name in cols:
#     temp = list()
#     for i in range(len(test)):
#         temp.append(test[header_name][i])
#     bigtemp.append(temp)
# 
# bigtemp = autoclean(bigtemp)
# j = 0
# for header_name in cols:
#     for i in range(len(test)):
#         test[header_name][i] = bigtemp[j][i]
#     j = j + 1

# In[6]:

for header_name in cols:
    temp = list()
    for i in range(len(test)):
        temp.append(test.iloc[i][header_name].astype(float))
    test[header_name]= temp
print(test.shape)
test


# In[7]:

train =train.dropna()
train.shape


# In[8]:

train2 = data[colRes].astype(int)
trainArr = train.as_matrix(cols) #training array
#trainArr = trainArr.replace(np.nan, 0, regex = True)
trainRes = train2.as_matrix(colRes) # training results
traindata=trainArr.astype(int)
traindata


# In[9]:


rf = RandomForestClassifier(n_estimators = 100) # initialize
rf.fit(trainArr, trainRes) # fit the data to the algorithm


# In[10]:



#test = test[cols]


# In[11]:

testArr = test.as_matrix(cols)
#testArr = testArr.replace(np.nan, 0, regex = True)
testArr


# In[12]:

test.shape


# In[13]:

results = rf.predict(testArr)
#results.toColumn()
results


# In[14]:

df = pd.read_csv('C:/Users/vbg22/Downloads/Machine learning Project/Tweeter-Bot-Detection/test_data_4_students.csv')
df2 = df['id']
df2 = pd.concat([df['id'],pd.DataFrame(results.T)], axis=1, join='inner')

df2.columns = ['Id','Bot']


# In[15]:

#df2.dropna()
df2['Id'].astype(int)
df2.to_csv('C:/Users/vbg22/Downloads/Machine learning Project/Tweeter-Bot-Detection/Output.csv', sep=',', index=False)
#df2['bot'] = rf.predict(testArr)
#np.shape(df2)
#test['bot'] = results
#test.head(10)


# In[16]:

#rf.score(test['bot'], test['predictions'])
Ans = pd.read_csv('C:/Users/vbg22/Downloads/Pictures/sub6.csv', encoding = 'UTF')
Pred = pd.read_csv('C:/Users/vbg22/Downloads/Machine learning Project/Tweeter-Bot-Detection/Output.csv', encoding = 'UTF')
#rf.score(Pred['Bot'], Ans['Bot'])


# In[17]:

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

len(Pred)
len(Ans)
Pred


# In[18]:

truePositive, falsePositive, trueNegative, falseNegative = calculateAccuracy(Ans['Bot'], Pred['Bot'], 575)
tpr = (truePositive) / (truePositive + falseNegative)
fpr = (falsePositive) / (falsePositive + trueNegative)

tpr = [0.0, tpr, 1.0]
fpr = [0.0, fpr, 1.0]


# In[19]:

"""
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
"""


# In[20]:

from sklearn.metrics import accuracy_score
accuracy_score(Ans['Bot'],Pred['Bot'])


# In[21]:

print(classification_report(Ans['Bot'], Pred['Bot']))


# In[22]:

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


# In[23]:

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


# In[24]:

print("Feature ranking:")

for f in range(train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, cols[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(train.shape[1]), indices)
plt.xlim([-1, train.shape[1]])
plt.show()


# In[25]:

for f in range(0,train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, cols[f], importances[indices[f]]))


# In[26]:

train.shape[1]

