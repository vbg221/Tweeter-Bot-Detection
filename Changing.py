# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:30:33 2017

@author: vbg22
"""

import numpy as np
import pandas as pd

fin = pd.read_csv('Output.csv', encoding='UTF')
fin['Id'].astype(np.int64)
#fin['Id'] = [(fin.iloc[i]['Id'].astype(int)) for i in range(len(fin))]
fin.to_csv('output1.csv', sep=',', index=False)
