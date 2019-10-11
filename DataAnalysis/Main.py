# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### inputs:
#DataPath = "C:/Users/Nadav/Documents/MATLAB/DataAnalysis/Data/test_data.csv"
DataPath = "C:/Users/naseg/Documents/MATLAB/DataAnalysis/Data/test_data.csv"
Q1 = '2019-07'
Q2 = '2019-04'
InputProduct = 'product_1'
SellsTH = 20 # [%]
Cross = np.array([['web','mobile_web',1],['mobile_web',1],['utm_65',1],\
                  ['utm_102',1],['medium_1',1],['medium_2',1],['medium_3',1]\
                  ,['medium_4',1]])

Cross2 = np.array(['web $ mobile_web','mobile_web','utm_65',\
                  'utm_102','medium_1','medium_2','medium_3'\
                  ,'medium_4'])

#### End inputs
Index = [Q1,Q2,'Trend[%]','Delta[%]']
Data = (pd.read_csv(DataPath)).round(1)

Qdates = Data.growth_policy_quote_level_created_quarter
ProductList = Data.Product
Data1 = Data[np.array(Qdates == Q1) & np.array(ProductList == InputProduct)]
Data2 = Data[np.array(Qdates == Q2) & np.array(ProductList == InputProduct)]

Delta = ((np.sum(Data1.new_revenue)/np.sum(Data2.new_revenue)-1)*100).round(1)
headers1 = list(Data1.columns.values)
headers2 = list(Data2.columns.values)

df = pd.DataFrame(columns = Cross2,index = Index)
IndexAPV = ['APV '+Q1,'APV '+Q2,'DeltaAPV[%]','DeltaPurchases[%]']
apv = pd.DataFrame(columns = ['Result'],index = IndexAPV)
#Col = []
for i in range(Cross.size):
    mask1 = np.zeros((Data1.shape), dtype=bool)
    mask2 = np.zeros((Data2.shape), dtype=bool)
    
    for j in range(np.size(Cross[i])-1):
        mask1 = np.logical_or(mask1,Data1[headers1].values == [Cross[i][j]])
        mask2 = np.logical_or(mask2,Data2[headers2].values == [Cross[i][j]])
        #Col[i] = Col[i] + Cross[i][j]
    ind1 = np.sum(mask1,1) == Cross[i][-1]
    ind2 = np.sum(mask2,1) == Cross[i][-1]

    df.set_value(Index[0],Cross2[i], (Data1[ind1].new_revenue.sum(axis = 0)).round(0))
    df.set_value(Index[1],Cross2[i], (Data2[ind2].new_revenue.sum(axis = 0)).round(0))
    df.set_value(Index[2],Cross2[i], ((df.loc[Index[1],Cross2[i]]/df.loc[Index[0],Cross2[i]]-1)\
                 *100).round(0))    
    df.set_value(Index[3],Cross2[i], ((df.loc[Index[1],Cross2[i]]-df.loc[Index[0],Cross2[i]])\
                 /(np.sum(Data1.new_revenue)-np.sum(Data2.new_revenue))*100).round(0)) 
    


#B= Data1[Data1.Product == 'product_1']
#a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
#pd.DataFrame(d1.values[mask], d1.index[mask], d1.columns)
#b = Data1.replace(to_replace ="utm_65",value =1)
#Data.columns
#Table1 = pd.pivot_table(Data, 'new_revenue',['Product'],aggfunc = np.sum,margins = True)
#Table1.values[Table1.index == 'product_1']
#Table2 = pd.pivot_table(Data,'new_revenue',['Product','Device'],aggfunc = np.sum,margins = True)
#Table3 = pd.pivot_table(Data1,'new_revenue',index = ['Product'],columns = ['Device'],aggfunc = np.sum,margins = True)

##### plots
ax = plt.subplot(111, frame_on=False) 
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)
plt.table(cellText=df.values,colWidths = [0.5]*len(df.columns),
          rowLabels=df.index,
          colLabels=df.columns,
          cellLoc = 'center', rowLoc = 'center',
          loc='top')
fig = plt.gcf()

print('Sells Change for: ' + InputProduct + \
      ' from ' + Q2 + ' to '+ Q1 + ' in ' +str(Delta) + '[%]')
if SellsTH < abs(Delta):
    print('This is significant change in sells')
else:
    print('This is not a significant change in sells')

apv.set_value(IndexAPV[0],'Result',  (np.sum(Data1.new_revenue)/np.sum(Data1.new_users)).round(0))
apv.set_value(IndexAPV[1],'Result',  (np.sum(Data2.new_revenue)/np.sum(Data2.new_users)).round(0))
apv.set_value(IndexAPV[2],'Result',  ((apv.at[IndexAPV[0],'Result']/apv.at[IndexAPV[1],'Result']-1)*100).round(0))
apv.set_value(IndexAPV[3],'Result',  ((np.sum(Data1.new_users)/np.sum(Data2.new_users)-1)*100).round(0))


