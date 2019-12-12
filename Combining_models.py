# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:58:38 2019

@author: Julien
"""

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from LDA import LDA
from logistic_regression import logReg
from DecisionTrees import DT
from SVM import SVM

import pandas as pd
import numpy as np

class Combined_Models():
    
    def __init__(self, X_train, T_train, method="DecisionTree"):
        if method=="SVM":
            self.clf = BaggingClassifier(base_estimator=SVC(C=0.1,kernel="linear",gamma="auto"),n_estimators=100, random_state=0)
            self.clf.fit(X_train, T_train)
        else :
            self.clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10, random_state=0)
            self.clf.fit(X_train, T_train)
            
    def prediction(self,X):
        return self.clf.predict(X)
                
    def error(self, X_train,T_train, X_test,T_test):
        err_train = self.clf.score(X_train,T_train)
        err_test = self.clf.score(X_test,T_test)
        return[err_train,err_test]
        
class Custom_Combined_Models():
    def __init__(self, X_train, T_train, cross_validation=False):
        self.cross_valid = cross_validation
        if not cross_validation :
            clf_SVC = SVC(C=0.1,kernel="linear",gamma="auto").fit(X_train, T_train)
            clf_LDA = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto').fit(X_train, T_train)
            clf_LR = LogisticRegression(solver='liblinear',C=0.4,multi_class='auto').fit(X_train, T_train)
            self.models=[clf_SVC,clf_LDA,clf_LR]       
            
        else :
            clf_SVC = SVM(X_train, T_train,True)
            #On n'utilise pas les données de test car on ne s'intéresse pas encore aux performances
            clf_LDA = LDA()
            clf_LDA.launch(X_train,X_train,T_train,T_train)
            clf_LR = logReg(X_train, T_train,True)
            self.models=[clf_SVC,clf_LDA,clf_LR]

    def prediction(self,X):
        results = []
        for model in self.models :
            results.append(model.predict(X))
                
        predictions = pd.DataFrame(results).mode(axis=0)
        return predictions.to_numpy()[0]
                
    def error(self, X_train,T_train, X_test,T_test, T_train_p, T_test_p):
        Err_train = np.int32(T_train_p!=T_train)
        Err_test = np.int32(T_test_p!=T_test)
        
        err_train = np.sum(Err_train)/len(X_train)*100
        err_test = np.sum(Err_test)/len(X_test)*100
        return[err_train,err_test]
        
    def predict_SVC(self,X):
        return self.models[0].predict(X)
    
    def predict_LDA(self,X):
        return self.models[1].predict(X)
    
    def predict_LR(self,X):
        return self.models[2].predict(X)
