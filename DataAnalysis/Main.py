# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# In[ ]: inputs:
#DataPath = "C:/Users/Nadav/Documents/MATLAB/DataAnalysis/Data/test_data.csv"
DataPath = "C:/Users/naseg/Documents/MATLAB/DataAnalysis/Data/test_data.csv"
Q1 = '2019-07'
Q2 = '2019-04'
InputProduct = 'product_2'
SellsTH = 20 # [%]
APV_TH = 5 # [%] (for stage 4)
purchases_TH = 5 # [%] (for stage 5)
DeltaPurchaseCVR_TH = 10 # [%] (for stage 8)
DeltaLeads_TH = 8 # [%] (for stage 9)

Cross = np.array([['web','mobile_web',1],['mobile_web',1],['utm_65',1],\
                  ['utm_102',1],['medium_1',1],['medium_2',1],['medium_3',1]\
                  ,['medium_4',1]])

Cross2 = np.array(['web $ mobile_web','mobile_web','utm_65',\
                  'utm_102','medium_1','medium_2','medium_3'\
                  ,'medium_4'])

# In[ ]: Pilar 1 - analyzing high level metrics
# In[ ]: step 1
Index = [Q1,Q2,'Trend[%]','Delta[%]']
Data = (pd.read_csv(DataPath)).round(1)

Qdates = Data.growth_policy_quote_level_created_quarter
ProductList = Data.Product
Data1 = Data[np.array(Qdates == Q1) & np.array(ProductList == InputProduct)]
Data2 = Data[np.array(Qdates == Q2) & np.array(ProductList == InputProduct)]

Delta = ((np.sum(Data1.new_revenue)/np.sum(Data2.new_revenue)-1)*100).round(1)
headers1 = list(Data1.columns.values)
headers2 = list(Data2.columns.values)

IndexAPV = ['APV '+Q1,'APV '+Q2,'DeltaAPV[%]','DeltaPurchases[%]'\
            ,'SellsCondition','APVCondition','PurchasesCondition'\
            ,'PurchaseCVR '+Q1,'PurchaseCVR '+Q2,'DeltaPurchaseCVR','DeltaLeads']
df = pd.DataFrame(columns = Cross2,index = Index)
apv = pd.DataFrame(columns = ['Result','Threshold','Analyse?'],index = IndexAPV)

apv.set_value(IndexAPV[0:4],'Analyse?', '')
apv.set_value(IndexAPV[0:4],'Threshold', '')
apv.set_value(IndexAPV[4],'Threshold', SellsTH)
apv.set_value(IndexAPV[5],'Threshold', APV_TH)
apv.set_value(IndexAPV[6],'Threshold', purchases_TH)
apv.set_value(IndexAPV[9],'Threshold', DeltaPurchaseCVR_TH)
apv.set_value(IndexAPV[10],'Threshold', DeltaLeads_TH)

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
    

# In[ ]:  plots
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
apv.set_value(IndexAPV[4],'Result', Delta)
if SellsTH < abs(Delta):
    print('This is significant change in sells')
    apv.set_value(IndexAPV[4],'Analyse?', 'Yes')
else:
    print('This is not a significant change in sells')
    apv.set_value(IndexAPV[4],'Analyse?', 'No')

# In[ ]: stage 3
apv.set_value(IndexAPV[0],'Result',  (np.sum(Data1.new_revenue)/np.sum(Data1.new_users)).round(0))
apv.set_value(IndexAPV[1],'Result',  (np.sum(Data2.new_revenue)/np.sum(Data2.new_users)).round(0))
apv.set_value(IndexAPV[2],'Result',  ((apv.at[IndexAPV[0],'Result']/apv.at[IndexAPV[1],'Result']-1)*100).round(0))
apv.set_value(IndexAPV[3],'Result',  ((np.sum(Data1.new_users)/np.sum(Data2.new_users)-1)*100).round(0))

# In[ ]: stage 4
AVPchange = abs((apv.at[IndexAPV[1],'Result']*np.sum(Data1.new_users)/\
        np.sum(Data1.new_revenue)*100).round(0) -100)
apv.set_value(IndexAPV[5],'Result', AVPchange)
if AVPchange > APV_TH:
    print(np.array2string(AVPchange) + '[%] This is significant change in APV')
    apv.set_value(IndexAPV[5],'Analyse?', 'Yes')
else:
    print(np.array2string(AVPchange) +'[%] This is not a significant change in APV')
    apv.set_value(IndexAPV[5],'Analyse?', 'No')

# In[ ]: stage 5,6
purchasesChange = abs((apv.at[IndexAPV[0],'Result']*np.sum(Data2.new_users)/\
        np.sum(Data1.new_revenue)*100).round(0) -100)
apv.set_value(IndexAPV[6],'Result', purchasesChange)
if purchasesChange > purchases_TH:
    print(np.array2string(purchasesChange) + '[%] This is significant change in APV')
    apv.set_value(IndexAPV[6],'Analyse?', 'Yes')
else:
    print(np.array2string(purchasesChange) +'[%] This is not significant change in purchases') 
    apv.set_value(IndexAPV[6],'Analyse?', 'No')

# In[ ]: stage 7
Purchase1 = np.sum(Data1.new_revenue)
Purchase2 = np.sum(Data2.new_revenue)
lead1 = np.sum(Data1.leads)
lead2 = np.sum(Data2.leads)

PurchaseCVR1 = (Purchase1/lead1).round(3)
PurchaseCVR2 = (Purchase2/lead2).round(3)
DeltaPurchaseCVR = ((PurchaseCVR2/PurchaseCVR1-1)*100).round(1)
DeltaLead = ((lead2/lead1-1)*100).round(1)

apv.set_value(IndexAPV[7],'Result', PurchaseCVR1)
apv.set_value(IndexAPV[8],'Result', PurchaseCVR2)
apv.set_value(IndexAPV[9],'Result', DeltaPurchaseCVR)
apv.set_value(IndexAPV[10],'Result', DeltaLead)

# In[ ]: stage 8
if abs(DeltaPurchaseCVR) > DeltaPurchaseCVR_TH:
    print(np.array2string(DeltaPurchaseCVR) + '[%] This is significant change in Delta Purchase CVR')
    apv.set_value(IndexAPV[9],'Analyse?', 'Yes')
else:
    print(np.array2string(DeltaPurchaseCVR) + '[%] This is not significant change in Delta Purchase CVR')
    apv.set_value(IndexAPV[9],'Analyse?', 'No')

# In[ ]: stage 9
if abs(DeltaLead) > DeltaLeads_TH:
    print(np.array2string(DeltaLead) + '[%] This is significant change in Delta Leads')
    apv.set_value(IndexAPV[10],'Analyse?', 'Yes')
else:
    print(np.array2string(DeltaLead) + '[%] This is not significant change in Delta Leads')
    apv.set_value(IndexAPV[10],'Analyse?', 'No')
# In[ ]:  plots
ax = plt.subplot(111, frame_on=False) 
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)
plt.table(cellText=apv.values,colWidths = [0.5]*len(apv.columns),
          rowLabels=apv.index,
          colLabels=apv.columns,
          cellLoc = 'center', rowLoc = 'center',
          loc='top')
fig = plt.gcf()

# In[ ]: Pilar 2 - dimenssions drill down
# In[ ]: stage 2



# In[ ]:  functions:
def Fields(Data,Str):
    #https://stackoverflow.com/questions/48851749/searching-for-string-in-all-columns-of-dataframe-in-python
    FloatData = Data.select_dtypes(exclude='object')
    BooleData = Data.select_dtypes(include='object')
    out = Data[Data.eq(Str).any(axis=1)]
    return (out)

FloatData = Data.select_dtypes(exclude='object')
BooleData = Data.select_dtypes(include='object')
List = list('s')
 L = []
for col in BooleData:
    tmp = BooleData[col].unique()
    List.append(tmp[:])
    #DataTmp = Fields(Data1,col)
    






