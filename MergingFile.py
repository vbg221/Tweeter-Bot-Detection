# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 08:57:51 2017

@author: vbg22
"""

import pandas as pd
import csv
import glob
import os

files = ["bots_data.csv","nonbots_data.csv"]

for filenames in files:
    df = pd.read_csv(filenames, encoding='latin1')
    df.to_csv('out.csv', mode='a')
