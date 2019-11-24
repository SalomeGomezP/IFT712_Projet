# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:58:14 2019

@author: Moi
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA:
    def __init__(self):
        self.discriminant=None

    def training(self, X, t):
        discriminant=LinearDiscriminantAnalysis(solver='svd')
        discriminant.fit(X, t)  
        self.discriminant=discriminant
        

    def prediction(self, x):
        t=self.discriminant.predict(x)
        return t
    
        
    def launch(self,x_train, x_test, t_train, t_test) :
        self.training(x_train, t_train)
        
        t_train_prediction=self.prediction(x_train)
        t_test_prediction=self.prediction(x_test)
        
        error_train=self.error(t_train, t_train_prediction)
        error_test=self.error(t_test, t_test_prediction)
        
        return[error_train,error_test]
        
    @staticmethod
    def error(t, prediction):
        difference=t-prediction
        number_error=np.count_nonzero(difference)
        number_error=number_error/len(t)           
        return number_error
