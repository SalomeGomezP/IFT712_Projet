# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:58:38 2019

@author: Julien
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


class Combined_Models():
    
    def __init__(self, X_train, T_train, method="DecisionTree"):
        if method=="SVM":
            self.clf = BaggingClassifier(base_estimator=SVC(C=0.1,kernel="linear",gamma="auto"),n_estimators=10, random_state=0)
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
        
