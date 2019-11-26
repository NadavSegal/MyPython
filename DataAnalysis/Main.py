import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

import DataInspection as DA

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
SalesShareTH = 105 # [%](for stage 5)
H_indexTH = 25 # [%](for stage 9)

# In[ ]: Pilar 1 - analyzing high level metrics
# In[ ]: step 1
apv,IndexAPV,Data,DataQ1,DataQ2 = DA.MatStage1(Q1,Q2,DataPath,InputProduct,SalesTH,APV_TH,purchases_TH,DeltaPurchaseCVR_TH,DeltaLeads_TH)

# In[ ]: stage 3
apv = DA.MatStage3(DataQ1,DataQ2,apv,IndexAPV)

# In[ ]: stage 4
apv = DA.MatStage4(apv,IndexAPV,DataQ1,APV_TH)

# In[ ]: stage 5,6
apv = DA.MatStage5_6(apv,IndexAPV,DataQ1,DataQ2,purchases_TH)

# In[ ]: stage 7
apv,DeltaPurchaseCVR,DeltaLead = DA.MatStage7(DataQ1,DataQ2,apv,IndexAPV)

# In[ ]: stage 8
apv = DA.MatStage8(DeltaPurchaseCVR,DeltaPurchaseCVR_TH,apv,IndexAPV)

# In[ ]: stage 9
apv = DA.MatStage9(DeltaLead,DeltaLeads_TH,apv,IndexAPV)

# In[ ]: Pilar 2 - dimenssions drill down
# In[ ]: stage 1
DimDrill,strData = DA.DimStage1(Data,DataQ1,DataQ2,Q1,Q2)

# In[]: stage 2
DimDrill = DA.DimStage2(DimDrill,strData,Q1,Q2,SumSellPosTH,SumSellNegTH)

# In[]: stage 3
DimDrill = DA.DimStage3(DimDrill,strData)

# In[]: stage 4: 
DimDrill = DA.DimStage4(DimDrill,strData,mTH,SumGrowthTH)     
DimDrill = DA.DimStage3(DimDrill,strData) #refill Growth-Rank

# In[]: stage 5:
DimDrill = DA.DimStage5(DimDrill,Q1,Q2,strData,SalesShareTH)

# In[]: stage 9:
DimDrill = DA.DimStage9(DimDrill,strData,H_indexTH)

# In[]: Data ordering
DimDrill = DimDrill.sort_values(['Metrica','IsPositiveGrowth','Growth-Rank'], ascending=[True, True, True])  




















 