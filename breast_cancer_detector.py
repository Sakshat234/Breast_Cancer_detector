# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 08:33:17 2020

@author: Suraj
"""
import pandas as pd #1234
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

df=pd.read_csv(r'C:\Users\Suraj\Desktop\os elab\breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print(accuracy)

example=np.array([4,2,2,1,3,2,1,1,1])
example=example.reshape(1,-1)
predict=clf.predict(example)
print(predict)
