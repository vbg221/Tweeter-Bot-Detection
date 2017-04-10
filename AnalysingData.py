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

sheet = pe.get_sheet(file_name='C:/Users/vbg22/Downloads/Machine Learning Project/out.csv', encoding = 'latin1')
sheet.column.select([3,4, 7, 8, 9, 11, 12, 13, 15, 19, 20]) # the column indices to keep
#cleaned_data = autoclean(sheet)
#datacleaner 'C:/Users/vbg22/Downloads/Machine Learning Project/out.csv' -o 'C:/Users/vbg22/Downloads/Machine Learning Project/out2.csv' -is , -os ,
sheet.save_as('C:/Users/vbg22/Downloads/Machine Learning Project/out1.csv')




#import pandas as pd

my_data = pd.read_csv('out1.csv', sep=',')
my_clean_data = autoclean(my_data)
my_data.to_csv('shity.csv', sep=',', index=False)

