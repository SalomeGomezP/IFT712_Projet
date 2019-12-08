# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:15:09 2019

@author: Moi
"""
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import cross_val_score
import numpy as np


class DT:
    def __init__(self):
        self.tree=None


    def choose_model(self, impurity_method, impurity_min):
        if(impurity_method=="") :
            return DecisionTreeClassifier()
        return DecisionTreeClassifier(criterion=impurity_method, min_impurity_decrease=impurity_min, max_depth=15)

    def training(self, X, t, impurity_method, impurity_min):
        tree=self.choose_model(impurity_method,impurity_min)
        tree.fit(X, t)  
        self.tree=tree
        

    def prediction(self, x):
        t=self.tree.predict(x)
        return t
    
        
    def launch(self,x_train, x_test, t_train, t_test, do_cross_validation, get_graph) :
        
        if (do_cross_validation==True):
            res_validation=self.cross_validation(x_train, t_train)
            self.training(x_train, t_train, res_validation[0],res_validation[1])
        else :
            self.training(x_train, t_train, "","")
        
        t_train_prediction=self.prediction(x_train)
        t_test_prediction=self.prediction(x_test)
        
        error_train=self.error(t_train, t_train_prediction)
        error_test=self.error(t_test, t_test_prediction)
        
        if(get_graph==True):
            export_graphviz(self.tree, out_file="tree.dot", filled=True, max_depth=5)

                    
        return[error_train,error_test]
        
    @staticmethod
    def error(t, prediction):
        difference=t-prediction
        number_error=np.count_nonzero(difference)
        number_error=number_error/len(t)           
        return number_error
    
    def cross_validation(self,x_train, t_train):
        k=4
        impurity_methods=["gini","entropy"]
        min_impurities=np.arange(0.0, 0.003,0.0001)
         
        error_best=0
        impurity_method_best=impurity_methods[0]
        min_impurity_best=0
        for i in range (len(impurity_methods)):
            print(len(min_impurities))
            for j in range(len(min_impurities)) :#min_impurity_decrease
                model=self.choose_model(impurity_methods[i],min_impurities[j])
                error_mean=cross_val_score(model, x_train, t_train, cv=k).mean()
                print(min_impurities[j])
                if (i==0 and j==0): #initialisation
                    error_best=error_mean
                    impurity_method_best=impurity_methods[i]
                    min_impurity_best=min_impurities[j]
                 
                if (error_mean<error_best):
                    error_best=error_mean
                    impurity_method_best=impurity_methods[i]
                    min_impurity_best=min_impurities[j]
                    
        print("Mesure d'impureté : "+ impurity_method_best)
        print("min_impurity_decrease : "+str(min_impurity_best))
        
        return [impurity_method_best,min_impurity_best]
                     
         
