# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:51:44 2020

@author: Serhat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%

df = pd.read_csv("iris.csv")
df.drop("Id", axis = 1, inplace = True
        )

#df.columns=['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species' ]

df= df[['Species','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','SepalLengthCm' ]] # change columns locations

ırıs_setosa =  df.loc[df.Species == "Iris-setosa"] 
ırıs_versicolor =  df.loc[df.Species == "Iris-versicolor"] 
ırıs_virginica =  df.loc[df.Species == "Iris-virginica"] 

#%% visualize

df.plot(kind ="box" , subplots = True, layout = (2,2), sharex = False , sharey = False)
df.hist()

from pandas.plotting import scatter_matrix
scatter_matrix(df)
plt.show()

