#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.unicode.east_asian_width', True)

# データの取得・加工
df = investpy.get_stock_historical_data(stock="GOOGL", country="united states", from_date="01/01/2010", to_date="01/01/2021")
df = df.resample("M").mean()
y = df["Close"]
y_lag = y.diff().dropna()



# ADF検定
from statsmodels.tsa.stattools import adfuller

adf,pvalue,usedlag,nobs,critical_values,icbest = adfuller(y_lag, autolag="AIC")
print(pd.DataFrame([["ADF統計量","P値","使用したラグ変数","臨界値とADF回帰に使用するデータ数","臨界値1%","臨界値5%","臨界値10%","情報基準の最大値"],
              [adf,pvalue,usedlag,nobs,critical_values["1%"],critical_values["5%"],critical_values["10%"],icbest]]).T)
#                                    0            1
# 0                          ADF統計量    -2.011163
# 1                                P値       0.2817
# 2                   使用したラグ変数           10
# 3  臨界値とADF回帰に使用するデータ数          120
# 4                           臨界値1%    -3.486056
# 5                           臨界値5%    -2.885943
# 6                          臨界値10%    -2.579785
# 7                   情報基準の最大値  1217.794854


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(endog = y, 
              order=(1, 1, 1))
result = model.fit()
print(result.summary())
#                                SARIMAX Results                                
# ==============================================================================
# Dep. Variable:                  Close   No. Observations:                  132
# Model:                 ARIMA(1, 1, 1)   Log Likelihood                -686.710
# Date:                Tue, 10 May 2022   AIC                           1379.420
# Time:                        06:48:54   BIC                           1388.046
# Sample:                    01-31-2010   HQIC                          1382.925
#                          - 12-31-2020                                         
# Covariance Type:                  opg                                         
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# ar.L1         -0.4824      0.136     -3.559      0.000      -0.748      -0.217
# ma.L1          0.7466      0.116      6.417      0.000       0.519       0.975
# sigma2      2089.0174    152.304     13.716      0.000    1790.508    2387.527
# ===================================================================================
# Ljung-Box (L1) (Q):                   0.55   Jarque-Bera (JB):               621.79
# Prob(Q):                              0.46   Prob(JB):                         0.00
# Heteroskedasticity (H):              16.75   Skew:                            -1.06
# Prob(H) (two-sided):                  0.00   Kurtosis:                        13.46
# ===================================================================================

# Warnings:
# [1] Covariance matrix calculated using the outer product of gradients (complex-step).

