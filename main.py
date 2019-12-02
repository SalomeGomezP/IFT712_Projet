# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:23:37 2019

@author: Julien
"""

import numpy as np
import pandas as pd
from LDA import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import & format
#data = pd.read_csv("data/train/train.csv")
#data= data.loc[data['Type'] == 1]
#msk = np.random.rand(len(data)) < 0.8
#data_train=data[msk]
#data_test=data[~msk]

data_train=pd.read_csv("train.csv")
data_test=pd.read_cvs("test.csv")

t_train=list(pd.Series(data_train['AdoptionSpeed'].values.tolist()))
t_test=list(pd.Series(data_test['AdoptionSpeed'].values.tolist()))




columns_to_drop=['Type','AdoptionSpeed','Name','Description','RescuerID','PetID']
data_train=data_train.drop(columns=columns_to_drop)
data_test=data_test.drop(columns=columns_to_drop)


##standardisation
#scaler = StandardScaler()
## Fit on training set only.
#scaler.fit(data_train)
## Apply transform to both the training set and the test set.
#data_train =pd.DataFrame(scaler.transform(data_train), columns=data_test.columns)
#data_test =pd.DataFrame(scaler.transform(data_test), columns=data_train.columns)

#ACP
#pca = PCA(.95)
#pca.fit(data_train)
#print(pca.n_components_)
#data_train = pca.transform(data_train)
#data_test = pca.transform(data_test)
#
#x_train=list(data_train)
#x_test=list(data_test)

x_train=list(pd.Series(data_train.values.tolist()))
x_test=list(pd.Series(data_test.values.tolist()))



LDA=LDA()
[err_train,err_test]=LDA.launch(x_train,x_test,t_train,t_test)

print('Erreur train = ', err_train, '%')
print('Erreur test = ', err_test, '%')



##normalisation données
#def normalisation_col(col) :
#    mean=col.mean()
#    std=col.std()
#    col=(col-mean)/std
#    return col
#    
#def normalisation(df) :
#    for column in df:
#        tmp=df[column]
#        df.loc[:,column]=normalisation_col(tmp)
#    return df
##    
#data_train=normalisation(data_train)
#data_test=normalisation(data_test)
