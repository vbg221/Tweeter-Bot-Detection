# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:35:45 2017

@author: vbg22
"""
from datacleaner import autoclean
import pandas as pd

my_data = pd.read_csv('testingtesting.csv', sep=',')
my_clean_data = autoclean(my_data)
my_data.to_csv('superawesome.csv', sep=',', index=False)
