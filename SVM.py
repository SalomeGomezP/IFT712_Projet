# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:01:14 2019

@author: Julien
"""

from sklearn.svm import SVC
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class SVM():
    
    def __init__(self,X_train, T_train, cross_val):
        if cross_val :
            self.cross_validation(X_train, T_train)
        else :
            self.clf = SVC(gamma='auto')
            self.clf.fit(X_train, T_train)
            
    def predict(self,X):
        return self.clf.predict(X)
        
    def cross_validation(self,X_train, T_train):
        err_min = 100 # en %
        c_best = 0.5
        k_best="rbf"
        deg_best=2
        gamma_best=0
        kernel=["linear", "rbf", "poly", "sigmoid"]
        c_array=np.logspace(-3, 5, 9)
        gamma_array=np.logspace(-5, 1, 6)

        for k in tqdm(kernel) :
            for c in c_array :
                if k!="linear":
                    for gamma in gamma_array :
                        if  k=='poly' :
                            for deg in range (1,5):
                                [err_min,c_best,k_best,deg_best,gamma_best] = self.test_parameters (X_train, T_train,c_best,k_best,deg_best,gamma_best,c,k,deg,gamma,err_min)
                else :
                    deg=0
                    gamma="auto"
                    [err_min,c_best,k_best,deg_best,gamma_best] = self.test_parameters (X_train, T_train,c_best,k_best,deg_best,gamma_best,c,k,deg,gamma,err_min)
                    
        self.clf = SVC(C=c_best,kernel=k_best).fit(X_train, T_train)
        #print("Paramètres optimaux : C= "+str(c_best)+"  Kernel= "+str(k_best)+" gamma= "+str(gamma_best))
        return
    
    def test_parameters(self,X_train, T_train,c_best,k_best,deg_best,gamma_best,c,k,deg,gamma,err_min):
        X_train2, X_valid, T_train2, T_valid = train_test_split(
        X_train, T_train, test_size=0.2, random_state=0)
        self.clf=SVC(C=c,kernel=k,degree=deg,gamma=gamma).fit(X_train2,T_train2)
        [err_train,err_valid]=self.error(X_train2,T_train2, X_valid,T_valid)
        if(err_valid<err_min):
            err_min=err_valid
            c_best = c
            k_best = k
            deg_best=deg
            gamma_best=gamma
        return [err_min,c_best,k_best,deg_best,gamma_best]
                
    def error(self, X_train,T_train, X_test,T_test):
        T_train_p = self.predict(X_train)
        self.T_train_p=T_train_p
        T_test_p = self.predict(X_test)
        Err_train = np.int32(T_train_p!=T_train)
        Err_test = np.int32(T_test_p!=T_test)
        
        err_train = np.sum(Err_train)/len(X_train)*100
        err_test = np.sum(Err_test)/len(X_test)*100
        return[err_train,err_test]
        
    