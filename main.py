# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:23:37 2019

@author: Julien
"""

import numpy as np
import pandas as pd
from LDA import LDA
from logistic_regression import logReg
from decision_trees import DT
from Combining_models import Combined_Models, Custom_Combined_Models
from SVM import SVM
from MLP import MLP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
# Import & format
# =============================================================================
data = pd.read_csv("train.csv")

#labelisation
data['species'] = data['species'].astype('category')
data['species']=data['species'].cat.codes

#separating data_train and data_test
msk = np.random.rand(len(data)) < 0.8
data_train=data[msk]
data_test=data[~msk]

t_train=list(pd.Series(data_train['species'].values.tolist()))
t_test=list(pd.Series(data_test['species'].values.tolist()))


# =============================================================================
# standardisation
# =============================================================================
scaler = StandardScaler()

scaler.fit(data_train)

data_train =pd.DataFrame(scaler.transform(data_train), columns=data_test.columns)
data_test =pd.DataFrame(scaler.transform(data_test), columns=data_train.columns)

# =============================================================================
# ACP
# =============================================================================
pca = PCA(.95)
pca.fit(data_train)
data_train = pca.transform(data_train)
data_test = pca.transform(data_test)

x_train=list(data_train)
x_test=list(data_test)



def launcher(method) :
    if (method=="LDA") :
        model=LDA()
        [err_train,err_test]=model.launch(x_train, x_test, t_train, t_test)
    if(method=="DT"):
        model=DT()
        [err_train,err_test]=model.launch(x_train,x_test,t_train,t_test, False, False)
    if (method=="MLP"):
        model=MLP()
        [err_train,err_test]=model.launch(x_train,x_test,t_train,t_test, False)
    if (method=="LR"):
        model=logReg(x_train,t_train,False)
        [err_train,err_test]=model.error(x_train,t_train,x_test,t_test)
    if (method=="SVM") :
        model = SVM(x_train,t_train,False)
        [err_train,err_test]=model.error(x_train,t_train,x_test,t_test)
    if(method=="combined models tree") :
        model = Combined_Models(x_train,t_train,"SVM")
        [err_train,err_test]=model.error(x_train,t_train,x_test,t_test)
    if(method=="combined models SVM") :
        model = Combined_Models(x_train,t_train,"SVM")
        [err_train,err_test]=model.error(x_train,t_train,x_test,t_test)
    if(method=="combined models all") :
        CCM = Custom_Combined_Models(x_train,t_train,False)
        T_train_p = CCM.prediction(x_train)
        T_test_p = CCM.prediction(x_test)
        [err_train,err_test] = CCM.error(x_train,t_train,x_test,t_test,T_train_p,T_test_p)

        
    print('Erreur train = ', err_train, '%')
    print('Erreur test = ', err_test, '%')
    
#launcher("LDA")
#launcher("DT")
#launcher("MLP")
#launcher("LR")
#launcher("SVM")
#launcher("combined models tree")
#launcher("combined models SVM")
launcher("combined models all")




