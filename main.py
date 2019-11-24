# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:23:37 2019

@author: Julien
"""

import numpy as np
import pandas as pd
from LDA import LDA

# Import & format
data = pd.read_csv("data/train/train.csv")
msk = np.random.rand(len(data)) < 0.8
data_train=data[msk]
data_test=data[~msk]

t_train=list(pd.Series(data_train['AdoptionSpeed'].values.tolist()))
columns_to_drop=['AdoptionSpeed','Name','Description','RescuerID','PetID']
x_train=list(pd.Series(data_train.drop(columns_to_drop,axis=1).values.tolist()))

t_test=list(pd.Series(data_test['AdoptionSpeed'].values.tolist()))
x_test=list(pd.Series(data_test.drop(columns_to_drop,axis=1).values.tolist()))


LDA=LDA()
[err_train,err_test]=LDA.launch(x_train,x_test,t_train,t_test)

print('Erreur train = ', err_train, '%')
print('Erreur test = ', err_test, '%')
