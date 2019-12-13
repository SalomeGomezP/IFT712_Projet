# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:03:12 2019

@author: Julien
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class logReg():
    
    def __init__(self,X_train, T_train, cross_val):
        if cross_val :
            self.cross_validation(X_train, T_train)
        else:
            self.clf = LogisticRegression(solver='liblinear',C=0.4,multi_class='auto').fit(X_train, T_train)
    
    def predict(self,X):
        return self.clf.predict(X)

    def error(self, X_train,T_train, X_test,T_test):
        T_train_p = self.predict(X_train)
        T_test_p = self.predict(X_test)
        Err_train = np.int32(T_train_p!=T_train)
        Err_test = np.int32(T_test_p!=T_test)
        
        err_train = np.sum(Err_train)/len(X_train)*100
        err_test = np.sum(Err_test)/len(X_test)*100
        return[err_train,err_test]
    
    def cross_validation(self,X_train, T_train):
        err_min = 100 # en %
        c_best=0.5
        t_best=1e-4
        
        for c_power in tqdm(range(30,100,5)):
            c=c_power/100
            for t in (1e-3,1e-4,1e-5):
                X_train2, X_valid, T_train2, T_valid = train_test_split(
                        X_train, T_train, test_size=0.2, random_state=0)
                
                self.clf = LogisticRegression(solver='liblinear',C=c,tol=t,multi_class='auto').fit(X_train2, T_train2)
                [err_train,err_test]=self.error(X_train2,T_train2, X_valid,T_valid)
                if(err_test<err_min):
                    err_min=err_test
                    c_best = c
                    t_best = t
        
        self.clf = LogisticRegression(solver='liblinear',C=c_best,tol=t_best,multi_class='auto').fit(X_train, T_train)
        #print("C= "+str(c_best)+"  tol= "+str(t_best))
        return   

