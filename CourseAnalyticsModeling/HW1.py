# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 07:32:51 2019

@author: Nadav
"""

# https://scikit-learn.org/stable/modules/svm.html
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC(gamma='scale')
clf.fit(X, y)  
A = clf.predict([[1., 2.]])