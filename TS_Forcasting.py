#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time series in Python:
    - Time series data preprocessing
    - Time series data analysis
    - Time series forecasting
*******************************************************************************
- To decompose a time series into components for further analysis:
Additive time series:
    value = base level + trend + seasonality + error
Multiplicative time series: (Not allowed when dataset contains 0 &/or - values)
    value = base level x trend x seasonality x error 
*******************************************************************************
Tip: Time series forecasting works only for a stationary time series since 
      only the behavior of a stationary time series is predictable.
      To make a time series stationary:
1. Remove non-regular behaviors that can change mean and/or covariance over time
2. Remove regular behaviors such as trend and seasonality that can change mean 
   and/or covariance over time
* A popular data transform method for removing non-stationary behaviors: Differencing
* ADF test (Augmented Dickey Fuller test):
    The default null hypothesis of the ADF test is that the time series is 
    non-stationary. When the p-value of the ADF test above is less than the 
    significance level of 0.05, we reject the null hypothesis and conclude that 
    the time series is stationary
* PACF (Partial Autocorrelation Function):
    By analyzing the results of it, the autoregressive order p can be determined
* ACF (Autocorrelation Function):
    The moving average order q can be determined by analyzing the results of it 
*******************************************************************************   
- An ARIMA model is determined by three parameters:
    p: the autoregressive order
    d: the order of differencing to make the time series stationary
    q: the moving average order
ARIMA model consists of three parts: 
    AutoRegression (AR), Moving Average (MA), and a constant
    ARIMA = constant + AR + MA
    where
AR = a linear combination of p consecutive values in the past time points 
MA = a linear combination of q consecutive forecast errors in the past time points
@author: khosravm
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## Loading data ===============================================================
df_raw = pd.read_csv('./datasets/GlobalTemp/GlobalTemperatures.csv', parse_dates=['dt'], index_col='dt')
print(df_raw.head())

# Extract the LandAverageTemperature column as a Pandas Series for demonstration purpose
df = df_raw['LandAverageTemperature']

# Handling Missing Data
print(df.isnull().value_counts())   # count no. of missing data

df = df.ffill() # use fill forward method for handling missing values

## Time Series Data Analysis ==================================================
# Visualizing Data
ax = df.plot(figsize=(16,5), title='Earth Surface Temperature')
ax.set_xlabel("Date")
ax.set_ylabel("Temperature")

# Decomposing Data into Components
""" extrapolate_trend='freq': To handle any missing values in the trend and 
    residuals at the beginning of the time series """
from statsmodels.tsa.seasonal import seasonal_decompose

additive    = seasonal_decompose(df, model='additive', extrapolate_trend='freq')
# To form a Pandas DataFrame from the resulting components of the additive decomposition 
additive_df = pd.concat([additive.seasonal, additive.trend, additive.resid, additive.observed], axis=1)
additive_df.columns = ['seasonal', 'trend', 'resid', 'actual_values']
print(additive_df.head())

# Visualize the additive decomposed components
plt.rcParams.update({'figure.figsize': (10,10)})
additive.plot().suptitle('Additive Decompose')

trend = additive.trend
ax = trend.plot(figsize=(16,5), title='Earth Surface Temperature')
ax.set_xlabel("Date")
ax.set_ylabel("Temperature")

## Time Series Forecasting ====================================================
#=============================== ARIMA ========================================
# Check if TS is stationary?!
from statsmodels.tsa.stattools import adfuller
result = adfuller(trend.values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# (1) Determining the order of differencing d:
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(trend.values); axes[0, 0].set_title('Original Series')
plot_acf(trend.values, ax=axes[0, 1]).suptitle('Original Series', fontsize=0)
# 1st Differencing (To make it stationary)
diff1 = trend.diff().dropna()
axes[1, 0].plot(diff1.values)
axes[1, 0].set_title('1st Order Differencing')
plot_acf(diff1.values, ax=axes[1, 1]).suptitle('1st Order Differencing', fontsize=0)
# 2nd Differencing
diff2 = trend.diff().diff().dropna()
axes[2, 0].plot(diff2.values)
axes[2, 0].set_title('2nd Order Differencing')
plot_acf(diff2.values, ax=axes[2, 1]).suptitle('2nd Order Differencing', fontsize=0)

"""=>> 2nd order of differencing does not make any improvement. 
       Thus the order of differencing d is set to 1 here. """

# (2) Determining the autoregressive order p
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
size = 100
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(diff1.values[:size])
axes[0].set_title('1st Order Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(diff1.values[:size], lags=50, ax=axes[1]).suptitle('1st Order Differencing', fontsize=0)

""" =>> The PACF lag 1 is well above the significance line (gray area). 
        Thus the autoregressive order p is set to 1."""
        
# (3) Determining the moving average order q
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
size = 100
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(diff1.values[:size])
axes[0].set_title('1st Order Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(diff1.values[:size], lags=50, ax=axes[1]).suptitle('1st Order Differencing', fontsize=0)

""" =>> The ACF lag 1 is well above the significance line (gray area). 
        Thus the moving average order q is set to 1. """
        
# Training ARIMA model
import statsmodels     
from statsmodels.tsa.arima_model import ARIMA
train = trend[:3000]
test  = trend[3000:]
# order = (p=1, d=1, q=1)
model = ARIMA(train, order=(1, 1, 1))  
model = model.fit(disp=0)  
print(model.summary())

# Plot residual errors
residuals = pd.DataFrame(model.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])

# Forecasting using the trained ARIMA model
# Forecast: 192 forecasting values with 95% confidence
fc, se, conf = model.forecast(192, alpha=0.05)
# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)

