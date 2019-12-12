# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:58:14 2019

@author: Moi
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

class LDA:
    def __init__(self):
        self.discriminant=None
        
    def choice_Solver(self, solver):
        if (solver =='svd') :
            return LinearDiscriminantAnalysis(solver='svd')
        if(solver=='lsqr') :
            return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        if (solver=='eigen' ) :
            return LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')

    def training(self, X, t, solver):
        discriminant=self.choice_Solver(solver)
        discriminant.fit(X, t)  
        self.discriminant=discriminant
        

    def predict(self, x):
        t=self.discriminant.predict(x)
        return t
    
        
    def launch(self,x_train, x_test, t_train, t_test) :
        #choosing solver via 3-cross validation
        solver=self.cross_validation(x_train, t_train)
        print("Solver sélectionné : "+solver)
        
        self.training(x_train, t_train, solver)
        
        t_train_prediction=self.predict(x_train)
        t_test_prediction=self.predict(x_test)
        
        error_train=self.error(t_train, t_train_prediction)
        error_test=self.error(t_test, t_test_prediction)
        
        return[error_train,error_test]
        
    @staticmethod
    def error(t, prediction):
        difference=t-prediction
        number_error=np.count_nonzero(difference)
        number_error=number_error/len(t)           
        return number_error
    
    def cross_validation(self,x_train, t_train):
         model=self.choice_Solver("svd")
         res_svd=cross_val_score(model, x_train, t_train, cv=3).mean()
         model=self.choice_Solver("lsqr")
         res_lsqr=cross_val_score(model, x_train, t_train, cv=3).mean()
         model=self.choice_Solver("eigen")
         res_eigen=cross_val_score(model, x_train, t_train, cv=3).mean()
         
         if (res_svd<=res_lsqr and res_svd<=res_eigen) :
             return 'svd'
         if(res_eigen<=res_lsqr and res_eigen<=res_svd):
             return 'eigen'
         if(res_lsqr<=res_eigen and res_lsqr<=res_svd):
             return 'lsqr'
         