# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:23:37 2019

@author: Julien
"""

import numpy as np
import pandas as pd
from logistic_regression import logReg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import & format
data = pd.read_csv("train.csv")

#labelisation
data['species'] = data['species'].astype('category')
data['species']=data['species'].cat.codes

msk = np.random.rand(len(data)) < 0.8
data_train=data[msk]
data_test=data[~msk]



t_train=list(pd.Series(data_train['species'].values.tolist()))
t_test=list(pd.Series(data_test['species'].values.tolist()))


##standardisation
scaler = StandardScaler()

scaler.fit(data_train)

data_train =pd.DataFrame(scaler.transform(data_train), columns=data_test.columns)
data_test =pd.DataFrame(scaler.transform(data_test), columns=data_train.columns)

#ACP
pca = PCA(.95)
pca.fit(data_train)
data_train = pca.transform(data_train)
data_test = pca.transform(data_test)

x_train=list(data_train)
x_test=list(data_test)

    
LR=logReg(x_train,t_train)
[err_train,err_test]=LR.train(x_train,t_train,x_test,t_test)

prediction_train = LR.T_train_p
print('Erreur train = ', err_train, '%')
print('Erreur test = ', err_test, '%')