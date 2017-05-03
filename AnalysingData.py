# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 09:16:08 2017

@author: vbg22
"""

import pandas as pd
import numpy as np
import pyexcel as pe

from datacleaner import autoclean
#names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
#df = pd.read_csv('C:/Users/vbg22/Downloads/Machine Learning Project/FinalData_edited.csv')
#df.head(6)
#f = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learningdatabases/'+'housing/housing.data',header=None,delim_whitespace=True,names=names,na_values='?')
#df.head(6)

sheet = pe.get_sheet(file_name='C:/Users/vbg22/Downloads/Machine Learning Project/training_data_2_csv_UTF.csv', encoding = 'latin1')
sheet.column.select([2,3, 6, 7, 8, 10, 11, 12, 14, 18, 19]) # the column indices to keep
#cleaned_data = autoclean(sheet)
#datacleaner 'C:/Users/vbg22/Downloads/Machine Learning Project/out.csv' -o 'C:/Users/vbg22/Downloads/Machine Learning Project/out2.csv' -is , -os ,
sheet.save_as('C:/Users/vbg22/Downloads/Machine Learning Project/training_data_2_csv_UTF1.csv')
df = pd.read_csv('C:/Users/vbg22/Downloads/Machine Learning Project/training_data_2_csv_UTF1.csv')

for i in range(0,len(df)):
    if(np.isnan(df['screen_name'][i])):
        df['screen_name'] = 0
    if(np.isnan(df['location'][i])):
        df['location'][i] = 0
    if(np.isnan(df['followers_count'][i])):
        df['followers_count'][i] = 0
    if(np.isnan(df['friends_count'][i])):
        df['friends_count'][i] = 0
    if(np.isnan(df['listedcount'][i])):
        df['listedcount'][i] = 0
    if(np.isnan(df['favourites_count'][i])):
        df['favourites_count'][i] = 0
    if(np.isnan(df['verified'][i])):
        df['verified'][i] = 0
    if(np.isnan(df['statuses_count'][i])):
        df['statuses_count'][i] = 0
    if(np.isnan(df['status'][i])):
        df['status'][i] = 0
    if(np.isnan(df['name'][i])):
        df['name'][i] = 0
    
    if(np.isnull(df['screen_name'][i])):
        df['screen_name'] = 0
    if(np.isnull(df['location'][i])):
        df['location'][i] = 0
    if(np.isnull(df['followers_count'][i])):
        df['followers_count'][i] = 0
    if(np.isnull(df['friends_count'][i])):
        df['friends_count'][i] = 0
    if(np.isnull(df['listedcount'][i])):
        df['listedcount'][i] = 0
    if(np.isnull(df['favourites_count'][i])):
        df['favourites_count'][i] = 0
    if(np.isnull(df['verified'][i])):
        df['verified'][i] = 0
    if(np.isnull(df['statuses_count'][i])):
        df['statuses_count'][i] = 0
    if(np.isnull(df['status'][i])):
        df['status'][i] = 0
    if(np.isnull(df['name'][i])):
        df['name'][i] = 0
    
df = df.dropna()    
df.to_csv('C:/Users/vbg22/Downloads/Machine Learning Project/out1.csv', sep=',', index=False)




#import pandas as pd

my_data = pd.read_csv('out1.csv', sep=',')
my_clean_data = autoclean(my_data)
my_data.to_csv('shity.csv', sep=',', index=False)

sheet2 = pe.get_sheet(file_name='C:/Users/vbg22/Downloads/Machine Learning Project/Tweeter-Bot-Detection/test_data_4_students.csv', encoding = 'UTF')
sheet2.column.select([2,3, 6, 7, 8, 10, 11, 12, 14, 18])
sheet2.save_as('C:/Users/vbg22/Downloads/Machine Learning Project/Tweeter-Bot-Detection/out2.csv')
my_data2 = pd.read_csv('C:/Users/vbg22/Downloads/Machine Learning Project/Tweeter-Bot-Detection/out2.csv', sep=',')
cleaned_test = autoclean(my_data2)
cleaned_test.to_csv('Final_test.csv', sep=',', index=False)
