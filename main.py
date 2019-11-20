# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:23:37 2019

@author: Julien
"""

import csv
import pandas as pd

# Import & format
data = pd.read_csv("data/train/train.csv")
dataT =data['AdoptionSpeed']
t_train=list(pd.Series(dataT.values.tolist()))
dataX=data.drop(['AdoptionSpeed','Name','Description','RescuerID','PetID'],axis=1)
x_train=list(pd.Series(dataX.values.tolist()))

#Visualization
for col in dataX :
    dataX.hist(column=col)