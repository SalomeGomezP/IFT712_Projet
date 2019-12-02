# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:03:12 2019

@author: Julien
"""

from sklearn.linear_model import LogisticRegression
import numpy as np

class logReg():
    
    def __init__(self,X_train, T_train):
        self.clf = LogisticRegression(C=0.5).fit(X_train, T_train)
        self.check1 = 1
    
    def predict(self,X):
        return self.clf.predict(X)

    def error(self, X_train,T_train, X_test,T_test):
        T_train_p = self.predict(X_train)
        self.T_train_p=T_train_p
        T_test_p = self.predict(X_test)
        Err_train = np.int32(T_train_p!=T_train)
        Err_test = np.int32(T_test_p!=T_test)
        
        err_train = np.sum(Err_train)/len(X_train)*100
        err_test = np.sum(Err_test)/len(X_test)*100
        return[err_train,err_test]
    
    