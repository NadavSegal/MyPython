#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:19:09 2019

@author: nadav
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def MatStage1(Q1,Q2,DataPath,InputProduct,SalesTH,APV_TH,purchases_TH,DeltaPurchaseCVR_TH,DeltaLeads_TH):
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
           
       return apv,IndexAPV, Data,Data1,Data2

def MatStage3(DataQ1,DataQ2,apv,IndexAPV):
       apv.set_value(IndexAPV[0],'Result',  (np.sum(DataQ1.new_revenue)/np.sum(DataQ1.new_users)).round(0))
       apv.set_value(IndexAPV[1],'Result',  (np.sum(DataQ2.new_revenue)/np.sum(DataQ2.new_users)).round(0))
       apv.set_value(IndexAPV[2],'Result',  ((apv.at[IndexAPV[0],'Result']/apv.at[IndexAPV[1],'Result']-1)*100).round(0))
       apv.set_value(IndexAPV[3],'Result',  ((np.sum(DataQ1.new_users)/np.sum(DataQ2.new_users)-1)*100).round(0))
       return apv

def MatStage4(apv,IndexAPV,DataQ1,APV_TH):
       AVPchange = abs((apv.at[IndexAPV[1],'Result']*np.sum(DataQ1.new_users)/\
       np.sum(DataQ1.new_revenue)*100).round(0) -100)
       apv.set_value(IndexAPV[5],'Result', AVPchange)
       if AVPchange > APV_TH:
           print(np.array2string(AVPchange) + '[%] This is significant change in APV')
           apv.set_value(IndexAPV[5],'Analyse?', 'Yes')
       else:
           print(np.array2string(AVPchange) +'[%] This is not a significant change in APV')
           apv.set_value(IndexAPV[5],'Analyse?', 'No')
       return apv

def MatStage5_6(apv,IndexAPV,DataQ1,DataQ2,purchases_TH):
       purchasesChange = abs((apv.at[IndexAPV[0],'Result']*np.sum(DataQ2.new_users)/\
               np.sum(DataQ1.new_revenue)*100).round(0) -100)
       apv.set_value(IndexAPV[6],'Result', purchasesChange)
       if purchasesChange > purchases_TH:
           print(np.array2string(purchasesChange) + '[%] This is significant change in APV')
           apv.set_value(IndexAPV[6],'Analyse?', 'Yes')
       else:
           print(np.array2string(purchasesChange) +'[%] This is not significant change in purchases') 
           apv.set_value(IndexAPV[6],'Analyse?', 'No')
       return apv        

def MatStage7(DataQ1,DataQ2,apv,IndexAPV):
       Purchase1 = np.sum(DataQ1.new_users)
       Purchase2 = np.sum(DataQ2.new_users)
       lead1 = np.sum(DataQ1.leads)
       lead2 = np.sum(DataQ2.leads)
       
       PurchaseCVR1 = (Purchase1/lead1).round(3)
       PurchaseCVR2 = (Purchase2/lead2).round(3)
       DeltaPurchaseCVR = ((PurchaseCVR2/PurchaseCVR1-1)*100).round(1)
       DeltaLead = ((lead2/lead1-1)*100).round(1)
       
       apv.set_value(IndexAPV[7],'Result', PurchaseCVR1)
       apv.set_value(IndexAPV[8],'Result', PurchaseCVR2)
       apv.set_value(IndexAPV[9],'Result', DeltaPurchaseCVR)
       apv.set_value(IndexAPV[10],'Result', DeltaLead) 
       return apv,DeltaPurchaseCVR,DeltaLead

def MatStage8(DeltaPurchaseCVR,DeltaPurchaseCVR_TH,apv,IndexAPV):
       if abs(DeltaPurchaseCVR) > DeltaPurchaseCVR_TH:
           print(np.array2string(DeltaPurchaseCVR) + '[%] This is significant change in Delta Purchase CVR')
           apv.set_value(IndexAPV[9],'Analyse?', 'Yes')
       else:
           print(np.array2string(DeltaPurchaseCVR) + '[%] This is not significant change in Delta Purchase CVR')
           apv.set_value(IndexAPV[9],'Analyse?', 'No')       
       return apv    

def MatStage9(DeltaLead,DeltaLeads_TH,apv,IndexAPV):
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
       return apv