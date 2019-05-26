# -*- coding: utf-8 -*-
"""
Created on Mon May 27 02:17:06 2019

@author: Abhishek_Sharma39
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 27 00:49:48 2019

@author: Abhishek_Sharma39
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def getMNIST():
    train = pd.read_csv(r"C:\Users\abhishek_sharma39\train.csv").values.astype(np.float32)
    train = shuffle(train)

    Xtrain = train[:-1000,1:] / 255
    Ytrain = train[:-1000,0].astype(np.int32)

    Xtest  = train[-1000:,1:] / 255
    Ytest  = train[-1000:,0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest
    

Xtrain, Ytrain, Xtest, Ytest = getMNIST()

covX = np.cov(Xtrain.T)

# Return the eigenvalues and eigenvectors: eigh
lambdas , Q = np.linalg.eigh(covX)

idx = np.argsort(-lambdas)
lambdas = lambdas[-idx]
lambdas = np.maximum(lambdas,0) # no negatives
Q = Q[:, idx]

Z = Xtrain.dot(Q)

plt.scatter(Z[:,0],Z[:,1], s =50, c = Ytrain, alpha = 0.5)
plt.show()

plt.plot(lambdas)
plt.title("Variance of each component")
plt.show()

# Return the cumulative sum of the elements along a given axis
plt.plot(np.cumsum(lambdas))
plt.title("Cumulative Variance")
plt.show()


     
    
    