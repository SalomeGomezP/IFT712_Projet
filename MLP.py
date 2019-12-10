# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:37:32 2019

@author: Moi
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class MLP:
    def __init__(self):
        self.mlp=None

    def training(self, X, t, params):
        mlp=MLPClassifier(params)
        mlp.fit(X, t)  
        self.mlp=mlp
        

    def prediction(self, x):
        t=self.mlp.predict(x)
        return t
    
        
    def launch(self,x_train, x_test, t_train, t_test) :
        
        params=self.search_hyperparams(x_train,t_train)
        
        self.training(x_train, t_train, params)
        
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
    
    def search_hyperparams(self,x,t):
        
        mlp = MLPClassifier(max_iter=100)
        
        #generation hidden layer(s)
        array=[]
        for  i in range (69,100,10):
            list_one_layer=list()          
            list_one_layer.append(i)
            array.append(tuple(list_one_layer))
            
            for j in range (69,100,10):
                list_two_layer=list()
                list_two_layer.append(i)
                list_two_layer.append(j)
                array.append(tuple(list_two_layer))
        
        parameter_space = {
        'hidden_layer_sizes': array,
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001,0.005, 0.001,0.05, 0.01],
        }
        
        search = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
        search.fit(x,t)
        
        print('Best parameters found:\n', search.best_params_)
        print('Loss associated :\n', search.best_score_)
        
        return search.best_params_