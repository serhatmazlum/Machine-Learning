# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:51:10 2020

@author: Serhat
"""

import pandas as pd
import numpy as np

# %%

data = pd.read_csv("Iris.csv")
data.drop("Id", axis = 1, inplace = True)

x,y = data.loc[:,data.columns != "Species"], data.loc[:,"Species"]

 #One hot encoder label
from sklearn.preprocessing import LabelEncoder
l_encoder= LabelEncoder()
y = l_encoder.fit_transform(y)

# Split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)

#KNN 

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
print("Knn score:",knn.score(x_test,y_test))

# Linear Regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x_train,y_train)
print("Linear Regression score:", linear_reg.score(x_test,y_test))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 42, max_iter=50)
lr.fit(x_train,y_train)
print("Logistic Regression score:", lr.score(x_test,y_test))

    
#%%