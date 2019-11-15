# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:23:37 2019

@author: Julien
"""

import csv
import pandas as pd

# Import
# =============================================================================
# t_train=[]
# x_train=[]
# with open("data/train.csv") as file:
#     reader = csv.reader(file)
#     titles = next(reader)
#     for row in reader :
#         t_train.append(row[1])
#         x_train.append([:1:])
# =============================================================================
data = pd.read_csv("data/train.csv")
dataT =data['Survived']
t_train=list(pd.Series(dataT.values.tolist()))
dataX=data.drop('Survived',axis=1)
x_train=list(pd.Series(dataX.values.tolist()))

#Visualization