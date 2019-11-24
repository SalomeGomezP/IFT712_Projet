# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:23:37 2019

@author: Julien
"""

import numpy as np
import pandas as pd
from logistic_regression import logReg

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

#Visualization
# =============================================================================
# for col in dataX :
#     dataX.hist(column=col)
# =============================================================================
    
LR=logReg(x_train,t_train)
[err_train,err_test]=LR.train(x_train,t_train,x_test,t_test)

prediction_train = LR.T_train_p
print('Erreur train = ', err_train, '%')
print('Erreur test = ', err_test, '%')