# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

# inputs:
DataPath = "C:/Users/Nadav/Documents/MATLAB/DataAnalysis/Data/test_data.csv"
Q1 = '2019-07'
Q2 = '2019-04'
#Cross = ['web'; 'mobile_web';'utm_65';'utm_102';'medium_1';'medium_2';'medium_3';'medium_4'];
Cross = np.array([['web'],['mobile_web'],['utm_65'],['utm_102'],['medium_1'],['medium_2'],['medium_3'],['medium_4']])

Data = pd.read_csv(DataPath)

Qdates = Data.growth_policy_quote_level_created_quarter
Data1 = Data[Qdates == Q1]
Data2 = Data[Qdates == Q2]
Delta = (np.sum(Data1.new_revenue)/np.sum(Data2.new_revenue)-1)*100
headers1 = list(Data1.columns.values)

for i in 1: len(Cross)
    1

mask1 = Data1[headers1].values == [Cross[0][0]]



B= Data1[Data1.Product == 'product_1']

#a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
#pd.DataFrame(d1.values[mask], d1.index[mask], d1.columns)
#b = Data1.replace(to_replace ="utm_65",value =1)
#Data.columns
Table1 = pd.pivot_table(Data, 'new_revenue',['Product'],aggfunc = np.sum,margins = True)
Table1.values[Table1.index == 'product_1']



Table2 = pd.pivot_table(Data,'new_revenue',['Product','Device'],aggfunc = np.sum,margins = True)
Table3 = pd.pivot_table(Data1,'new_revenue',index = ['Product'],columns = ['Device'],aggfunc = np.sum,margins = True)

