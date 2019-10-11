# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:16:20 2019

@author: Nadav
"""
import wget
#! wget https://raw.githubusercontent.com/hse-aml/intro-to-dl/master/setup_google_colab.py -O setup_google_colab.py
Out = 'C:/Users/Nadav/Documents/MyPython/AdvancedML'
wget.download('https://raw.githubusercontent.com/hse-aml/intro-to-dl/master/setup_google_colab.py',\
              out = Out)
import setup_google_colab
setup_google_colab.setup_week1()  # change to the week you're working on
# note on week 2: select setup_week2_v2() if you've started the course after August 13, 2018,
# otherwise call setup_week2().

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import grading
grader = grading.Grader(assignment_key="UaHtvpEFEee0XQ6wjK-hZg", 
                      all_parts=["xU7U4", "HyTF6", "uNidL", "ToK7N", "GBdgZ", "dLdHG"])


with open('train.npy', 'rb') as fin:
    X = np.load(fin)