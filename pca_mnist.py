# -*- coding: utf-8 -*-
"""
Created on Mon May 27 00:49:48 2019

@author: Abhishek_Sharma39
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def getMNIST():
    train = pd.read_csv(r"C:\Users\abhishek_sharma39\train.csv").values.astype(np.float32)
    train = shuffle(train)
    
    train = shuffle(train)

    Xtrain = train[:-1000,1:] / 255
    Ytrain = train[:-1000,0].astype(np.int32)

    Xtest  = train[-1000:,1:] / 255
    Ytest  = train[-1000:,0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest
    
    

def main():
    Xtrain, Ytrain, Xtest, Ytest = getMNIST()
    
    pca = PCA()
    reduced_pca = pca.fit_transform(Xtrain)
    plt.scatter(reduced_pca[:,0],reduced_pca[:,1],s= 50, c= Ytrain, alpha = 0.3)
    plt.show()
     
    plt.plot(reduced_pca.explained_variance_ratio)
    plt.show()
    
    cum = []
    value = 0
    for var in reduced_pca.explained_variance_ratio:
        cum.append(value+var)
        
    plt.plot(cum)
    plt.show()
    
if __name__ == "__main__":
    main()
     
     
    
    