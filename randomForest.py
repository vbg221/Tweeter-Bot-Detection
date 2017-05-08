# coding: utf-8
"""
Created on Wed Apr  5 20:30:35 2017

This program uses Random Forest Algorithm to predict whether or not a twitter account is bot.
It reads data from a filtereed CSV file.

@author: vbg22
"""


import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datacleaner import autoclean

'''
Read Training and Testing data from file.
'''
data = pd.read_csv('C:/Users/vbg22/Downloads/Pictures/binary and continouous.csv') #, encoding = 'latin1'
data2 = pd.read_csv('C:/Users/vbg22/Downloads/Machine learning Project/Tweeter-Bot-Detection/test2.csv', encoding = 'latin1')

'''
Drop the lines that may contain NaN values.
'''
data = data.dropna()
data2.dropna()

'''
Replace nan values with null is they still exists in data frame after using dropna()
'''
data = data.replace(np.nan, '', regex = True)
data2 = data2.replace(np.nan, '', regex = True)

'''
Define features from the data that are actually important.
cols contains the features to be selected from training dataframe.
colRes contains the labels of each tupple.
'''
#,'url','created_at','lang','status','favourites_count'
cols = ['screen_name','location','description','friends_count','followers_count','listedcount']
colRes = ['bot']                    

'''
Training and Testing data is gathered for only desired features.                 
Unnecessary features are ignored.
'''
train = data
test = data2
train = train[cols]
test = test[cols]
test

'''
For the both training and testing dataframe, each field is converted into float value.
'''
for header_name in cols:
    temp = list()
    for i in range(len(train)):
        temp.append(train.iloc[i][header_name].astype(float))
    train[header_name]= temp

for header_name in cols:
    temp = list()
    for i in range(len(test)):
        temp.append(test.iloc[i][header_name].astype(float))
    test[header_name]= temp

train =train.dropna()
train2 = data[colRes].astype(int)


'''
Training data and labels are converted to matrices so that random forest can be
fitted on the training data.
'''

trainArr = train.as_matrix(cols) #training array
#trainArr = trainArr.replace(np.nan, 0, regex = True)
trainRes = train2.as_matrix(colRes) # training results
traindata=trainArr.astype(int)

#Used estimators value = 100 which is default but the model gives same results
#even when the estimators value is down to 20. Increasing it does not make much 
#difference to the final outcome due to low number of features.
rf = RandomForestClassifier(n_estimators = 100) # initialize
rf.fit(trainArr, trainRes) # fit the data to the algorithm

'''
Test data is converted to matrix so that Random Forest can be used to predict 
its labels.
'''
testArr = test.as_matrix(cols)
results = rf.predict(testArr)
results

'''
Append the results with ID of the account and save it to CSV.
'''
df = pd.read_csv('C:/Users/vbg22/Downloads/Machine learning Project/Tweeter-Bot-Detection/test_data_4_students.csv')
df2 = df['id']
df2 = pd.concat([df['id'],pd.DataFrame(results.T)], axis=1, join='inner')
df2.columns = ['Id','Bot']
df2['Id'].astype(int)
df2.to_csv('C:/Users/vbg22/Downloads/Machine learning Project/Tweeter-Bot-Detection/Output.csv', sep=',', index=False)



"""
The next section is for calculating accuracy and identyfy the significance of
each feature.
"""

'''
Read the CSV files to data frames.
'''
Ans = pd.read_csv('C:/Users/vbg22/Downloads/Pictures/sub6.csv', encoding = 'UTF')
Pred = pd.read_csv('C:/Users/vbg22/Downloads/Machine learning Project/Tweeter-Bot-Detection/Output.csv', encoding = 'UTF')

'''
Define the function that calculates accuracy.
Arguements  : (bot, predictions, length)
              bot : original labeled data that identifies account as bot.
              predictions : predicted labels for the same accounts.
              length : length of the dataframe (Total count of accounts)
Output      : [tp,fp,tn,fn]
              tp : number of True Positives
              fp : number of False Positives
              tn : number of True Negatives
              fn : number of False Negatives
'''
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

'''
Calculatin accuracy and true positive and false positive rates.
'''
truePositive, falsePositive, trueNegative, falseNegative = calculateAccuracy(Ans['Bot'], Pred['Bot'], 575)
tpr = (truePositive) / (truePositive + falseNegative)
fpr = (falsePositive) / (falsePositive + trueNegative)

tpr = [0.0, tpr, 1.0]
fpr = [0.0, fpr, 1.0]

'''
Plot an ROC to show the statstics of the results.
'''
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

'''
Inherent accuracy of the model.
'''
from sklearn.metrics import accuracy_score
accuracy_score(Ans['Bot'],Pred['Bot'])

#Classification Report that shows Precision, Recall, f1 score, support
print(classification_report(Ans['Bot'], Pred['Bot']))

'''
Analyses of the significance of features that were used.
'''
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

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

for f in range(0,train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, cols[indices[f]], importances[indices[f]]))

#train.shape[1]

