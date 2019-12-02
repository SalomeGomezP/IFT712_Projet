# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:15:09 2019

@author: Moi
"""
from sklearn.tree import DecisionTreeClassifier

class DT:
    def __init__(self):
        self.tree=None

    def training(self, X, t):
        tree=DecisionTreeClassifier()
        tree.fit(X, t)  
        self.tree=tree
        

    def prediction(self, x):
        t=self.tree.predict(x)
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