# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 01:29:48 2017

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

plt.rcParams['figure.figsize'] = (12.0, 6.0)

data = pd.read_csv(r'C:\Users\User\Downloads\Advanced Regression\train.csv')

prices = pd.DataFrame({"price":data["SalePrice"], "log(price + 1)":np.log1p(data["SalePrice"])})

data['SalePrice'] = np.log1p(data['SalePrice'])

numeric_feat = data.dtypes[data.dtypes != 'object'].index

skewed_feat = data[numeric_feat].apply(lambda x: skew(x.dropna()))

skewed_feat = skewed_feat[skewed_feat > 0.75]

skewed_feat = skewed_feat.index

data[skewed_feat] = np.log1p(data[skewed_feat])



