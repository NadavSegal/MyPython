# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
# In[ ]: inputs:
#DataPath = "C:/Users/Nadav/Documents/MATLAB/DataAnalysis/Data/test_data.csv"
DataPath = "/home/nadav/Documents/MATLAB/DataAnalysis/Data/test_data.csv"
Q1 = '2019-07'
Q2 = '2019-04'
InputProduct = 'product_2'
# Pilar 1:
SalesTH = 20 # [%]
APV_TH = 5 # [%] (for stage 4)
purchases_TH = 5 # [%] (for stage 5)
DeltaPurchaseCVR_TH = 10 # [%] (for stage 8)
DeltaLeads_TH = 8 # [%] (for stage 9)
# pilar 2:
SumSellPosTH = 85 # [%] (for stage 3)
SumSellNegTH = 85 # [%] (for stage 3)
mTH = 7 # (for stage 4)
SumGrowthTH = 0.7  # (for stage 4)

# In[ ]: Pilar 1 - analyzing high level metrics
# In[ ]: step 1
Index = [Q1,Q2,'Trend[%]','Delta[%]']
Data = (pd.read_csv(DataPath)).round(1)
Data = Data.fillna('')

Qdates = Data.growth_policy_quote_level_created_quarter
ProductList = Data.Product
Data1 = Data[np.array(Qdates == Q1) & np.array(ProductList == InputProduct)]
Data2 = Data[np.array(Qdates == Q2) & np.array(ProductList == InputProduct)]

Delta = ((np.sum(Data1.new_revenue)/np.sum(Data2.new_revenue)-1)*100).round(1)
headers1 = list(Data1.columns.values)
headers2 = list(Data2.columns.values)

IndexAPV = ['APV '+Q1,'APV '+Q2,'DeltaAPV[%]','DeltaPurchases[%]'\
            ,'SalesCondition','APVCondition','PurchasesCondition'\
            ,'PurchaseCVR '+Q1,'PurchaseCVR '+Q2,'DeltaPurchaseCVR','DeltaLeads']

apv = pd.DataFrame(columns = ['Result','Threshold','Analyse?'],index = IndexAPV)

apv.set_value(IndexAPV[0:4],'Analyse?', '')
apv.set_value(IndexAPV[0:4],'Threshold', '')
apv.set_value(IndexAPV[4],'Threshold', SalesTH)
apv.set_value(IndexAPV[5],'Threshold', APV_TH)
apv.set_value(IndexAPV[6],'Threshold', purchases_TH)
apv.set_value(IndexAPV[9],'Threshold', DeltaPurchaseCVR_TH)
apv.set_value(IndexAPV[10],'Threshold', DeltaLeads_TH)

# In[ ]:  plots

print('Sales Change for: ' + InputProduct + \
      ' from ' + Q2 + ' to '+ Q1 + ' in ' +str(Delta) + '[%]')
apv.set_value(IndexAPV[4],'Result', Delta)
if SalesTH < abs(Delta):
    print('This is significant change in sales')
    apv.set_value(IndexAPV[4],'Analyse?', 'Yes')
else:
    print('This is not a significant change in sales')
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
Purchase1 = np.sum(Data1.new_users)
Purchase2 = np.sum(Data2.new_users)
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
FloatData = Data.select_dtypes(exclude='object')
strData = Data.select_dtypes(include='object')
#DimDrill = pd.DataFrame(columns = ['Sales','Leads','Purcheses'],index = ['te1','te2'])
DimDrill = pd.DataFrame()
j = 0
jj = 0
fields = np.empty(strData.columns.shape[0], dtype=object)
for col in strData[strData.columns[[0,1,2,3,5,8]]]:
    fields[j] = strData[col].unique()
    for field in fields[j]:
        DimDrill.set_value(field,'Metrica',col)
        # Data1:
        SumSales1 = np.sum(Data1[Data1[col].str.match(field)].new_revenue) 
        SumLead1 = np.sum(Data1[Data1[col].str.match(field)].leads)         
        SumPurchase1 = np.sum(Data1[Data1[col].str.match(field)].new_users)                 
        DimDrill.set_value(field, Q1+' Sales',SumSales1)
        DimDrill.set_value(field, Q1+' Leads',SumLead1)
        DimDrill.set_value(field, Q1+' Purcheses',SumPurchase1)        
        # Data2:
        SumSales2 = np.sum(Data2[Data2[col].str.match(field)].new_revenue) 
        SumLead2 = np.sum(Data2[Data2[col].str.match(field)].leads)         
        SumPurchase2 = np.sum(Data2[Data2[col].str.match(field)].new_users) 
        DimDrill.set_value(field, Q2+' Sales',SumSales2)
        DimDrill.set_value(field, Q2+' Leads',SumLead2)
        DimDrill.set_value(field, Q2+' Purcheses',SumPurchase2)        
        # common:
        DimDrill.set_value(field,'IsPositiveGrowth',SumSales2>SumSales1)  
        
        jj = jj+1
    j =j+1

# In[]: stage 2
for col in strData[strData.columns[[0,1,2,3,5,8]]]:
        PosIndx = np.logical_and(DimDrill.IsPositiveGrowth,DimDrill.Metrica == col)
        NegIndx = np.logical_and(~DimDrill.IsPositiveGrowth,DimDrill.Metrica == col)
        # Data1:
        SumPos1 = np.sum(DimDrill[Q1+' Sales'][PosIndx])
        SumNeg1 = np.sum(DimDrill[Q1+' Sales'][NegIndx])
        # Data2:
        SumPos2 = np.sum(DimDrill[Q2+' Sales'][PosIndx])
        SumNeg2 = np.sum(DimDrill[Q2+' Sales'][NegIndx])
        fields = strData[col].unique()
        # Common:
        for field in fields:
               PosFieldInd = np.logical_and(PosIndx,DimDrill.index == field)
               NegFieldInd = np.logical_and(NegIndx,DimDrill.index == field)
               PosGrowth = np.sum(DimDrill[Q2+' Sales'][PosFieldInd]-DimDrill[Q1+' Sales'][PosFieldInd])/(SumPos2 - SumPos1)
               NegGrowth = np.sum(DimDrill[Q2+' Sales'][NegFieldInd]-DimDrill[Q1+' Sales'][NegFieldInd])/(SumNeg2 - SumNeg1)
               DimDrill.set_value(DimDrill[PosFieldInd].index,'Growth,Pos/Neg' ,PosGrowth)
               DimDrill.set_value(DimDrill[NegFieldInd].index,'Growth,Pos/Neg' ,NegGrowth)
        # stage 3
        if SumPos1/(SumPos1+SumNeg1)*100 > SumSellPosTH:               
               DimDrill.set_value(PosIndx,'Analyse' ,'YES')
               DimDrill.set_value(NegIndx,'Analyse' ,'NO')
        elif SumNeg1/(SumPos1+SumNeg1)*100 > SumSellNegTH:
               DimDrill.set_value(PosIndx,'Analyse' ,'NO')
               DimDrill.set_value(NegIndx,'Analyse' ,'YES')
        else:
               DimDrill.set_value(np.logical_or(NegIndx,PosIndx),'Analyse' ,'YES')

# In[]: stage 3
#  Positive:
for col in strData[strData.columns[[0,1,2,3,5,8]]]:
        PosIndx = np.logical_and(DimDrill.IsPositiveGrowth,DimDrill.Metrica == col)
        fields = DimDrill['Growth,Pos/Neg'][PosIndx]
        fields = fields.sort_values(ascending=False)
        fields = fields.index
        j=1
        for field in fields:
               PosFieldInd = np.logical_and(PosIndx,DimDrill.index == field)
               DimDrill.set_value(DimDrill[PosFieldInd].index,'Growth-Rank' ,j)
               j = j+1
#  Negative:
for col in strData[strData.columns[[0,1,2,3,5,8]]]:
        NegIndx = np.logical_and(~DimDrill.IsPositiveGrowth,DimDrill.Metrica == col)
        fields = DimDrill['Growth,Pos/Neg'][NegIndx]
        fields = fields.sort_values(ascending=False)
        fields = fields.index
        j=1
        for field in fields:
               NegFieldInd = np.logical_and(NegIndx,DimDrill.index == field)
               DimDrill.set_value(DimDrill[NegFieldInd].index,'Growth-Rank' ,j)
               j = j+1
# Remove empty un changed Growth fields: 
ind = DimDrill['Growth,Pos/Neg'] == 0            
DimDrill = DimDrill.drop(DimDrill.index[ind],axis = 0)
DimDrill = DimDrill.sort_values(['Metrica','IsPositiveGrowth','Growth-Rank'], ascending=[True, True, True])

# stage 4: 
# if sum(Growth_Share_fieldXm)>=70% where 1<m<=7 then group by m
# positive:
for col in strData[strData.columns[[0,1,2,3,5,8]]]:
       NegIndx = np.logical_and(~DimDrill.IsPositiveGrowth,DimDrill.Metrica == col)
       SumGrowth = 0
       OtherNegIndx = ~NegIndx == NegIndx
       for m in range(1,np.sum(NegIndx == True)+1):              
              NegIndxM = np.logical_and(NegIndx,DimDrill['Growth-Rank'] ==m)
              SumGrowth = SumGrowth + (np.sum(DimDrill['Growth,Pos/Neg'][NegIndxM]))
              if m < mTH and SumGrowth < SumGrowthTH:         
                     DimDrill.set_value(NegIndxM,'Gruop' ,['G.Neg-' + np.str(m)])
                     mFinal = m
              else:  
                     OtherNegIndx[NegIndxM] = True              
       
       if OtherNegIndx.any():
              tmp = DimDrill[OtherNegIndx].sum(axis=0)
              DimDrill = DimDrill.drop(DimDrill.index[OtherNegIndx],axis = 0)
              DimDrill.loc[col + '-Other-Neg'] = tmp
              DimDrill.set_value(col + '-Other-Neg','Gruop' ,'G.Neg-'+ np.str(mFinal+1)+' Batch')
              DimDrill.set_value(col + '-Other-Neg','Growth-Rank' , (mFinal+1))
              DimDrill.set_value(col + '-Other-Neg','Metrica' , col)
              
       PosIndx = np.logical_and(DimDrill.IsPositiveGrowth,DimDrill.Metrica == col)
       SumGrowth = 0
       OtherPosIndx = ~PosIndx == PosIndx
       for m in range(1,np.sum(PosIndx == True)+1):              
              PosIndxM = np.logical_and(PosIndx,DimDrill['Growth-Rank'] ==m)
              SumGrowth = SumGrowth + (np.sum(DimDrill['Growth,Pos/Neg'][PosIndxM]))
              if m < mTH and SumGrowth < SumGrowthTH:         
                     DimDrill.set_value(PosIndxM,'Gruop' ,['G.Pos-' + np.str(m)])
                     mFinal = m
              else:  
                     OtherPosIndx[PosIndxM] = True              
       
       if OtherPosIndx.any():
              tmp = DimDrill[OtherPosIndx].sum(axis=0)
              DimDrill = DimDrill.drop(DimDrill.index[OtherPosIndx],axis = 0)
              DimDrill.loc[col + '-Other-Pos'] = tmp
              DimDrill.set_value(col + '-Other-Pos','Gruop' ,'G.Pos-'+ np.str(mFinal+1)+' Batch')
              DimDrill.set_value(col + '-Other-Pos','Growth-Rank' , (mFinal+1))
              DimDrill.set_value(col + '-Other-Pos','Metrica' , col)       
       
DimDrill = DimDrill.sort_values(['Metrica','IsPositiveGrowth','Growth-Rank'], ascending=[True, True, True])       