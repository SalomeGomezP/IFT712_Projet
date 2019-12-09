# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:01:14 2019

@author: Julien
"""

from sklearn.svm import SVC
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#le jeu de données étant de petite taille, on peu utiliser le SVC

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
        kernel=["linear", "rbf", "poly", "sigmoid"]
        for k in tqdm(kernel) :
            for c_power in range(30,100,5):
                c=c_power/100                
                if  k=='poly' :
                    for deg in range (1,5):
                        self.test_parameters (X_train, T_train,c_best,k_best,deg_best,c,k,deg,err_min)
                else :
                    deg=0
                    self.test_parameters (X_train, T_train,c_best,k_best,deg_best,c,k,deg,err_min)
                    
        self.clf = SVC(C=c_best,kernel=k_best).fit(X_train, T_train)
        print("C= "+str(c_best)+"  Kernel= "+str(k_best))
        return
    
    def test_parameters(self,X_train, T_train,c_best,k_best,deg_best,c,k,deg,err_min):
        X_train2, X_valid, T_train2, T_valid = train_test_split(
        X_train, T_train, test_size=0.2, random_state=0)
        self.clf=SVC(gamma='auto',C=c,kernel=k,degree=deg).fit(X_train2,T_train2)
        [err_train,err_test]=self.error(X_train2,T_train2, X_valid,T_valid)
        if(err_test<err_min):
            err_min=err_test
            c_best = c
            k_best = k
            deg_best=deg
        return [c_best,k_best,deg_best]
                
    def error(self, X_train,T_train, X_test,T_test):
        T_train_p = self.predict(X_train)
        self.T_train_p=T_train_p
        T_test_p = self.predict(X_test)
        Err_train = np.int32(T_train_p!=T_train)
        Err_test = np.int32(T_test_p!=T_test)
        
        err_train = np.sum(Err_train)/len(X_train)*100
        err_test = np.sum(Err_test)/len(X_test)*100
        return[err_train,err_test]