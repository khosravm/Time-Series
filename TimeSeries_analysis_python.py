#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:00:22 2020
Time Series Analysis in Python 
https://www.machinelearningplus.com/time-series/time-series-analysis-python/
https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
@author: Mahdieh Khosravi
"""
from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

# Import TS as Dataframe
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
print('----------------------')
print('   Dataframe head ')
print('----------------------')
print(df.head())
print('----------------------')

# Import TS as a Pandas Series 
ser = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')
print('\n')
print('----------------------')
print(' Pandas Series head ')
print('----------------------')
print(ser.head())
print('----------------------')

# Panal Data
"""
Panel data is also a time based dataset.
The difference is that, in addition to time series, it also contains one or 
more related variables that are measured for the same time periods.
"""

# dataset source: https://github.com/rouseguy
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/MarketArrivals.csv')
df = df.loc[df.market=='MUMBAI', :]
#df = df.loc[df.month=='June', :]
print(df.head())

### Visualizing a time series #################################################
# Time series data source: fpp pacakge in R.
import matplotlib.pyplot as plt
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')

# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.index, y=df.value, title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.')    

## hline = Since all values are positive, you can show this 
# on both sides of the Y axis to emphasize the growth
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
x  = df['date'].values
y1 = df['value'].values

# Plot
fig, ax = plt.subplots(1, 1, figsize=(16,5), dpi= 120)
plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5, linewidth=2, color='seagreen')
plt.ylim(-100, 100)
plt.title('Air Passengers (Two Side View)', fontsize=16)
plt.hlines(y=0, xmin=np.min(df.date), xmax=np.max(df.date), linewidth=.5)
plt.show()

# Seasonal Plot of a Time Series
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')
df.reset_index(inplace=True)

# Prepare data
df['year'] = [d.year for d in df.date]
df['month'] = [d.strftime('%b') for d in df.date]
years = df['year'].unique()

# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(16,12), dpi= 80)
for i, y in enumerate(years):
    if i > 0:        
        plt.plot('month', 'value', data=df.loc[df.year==y, :], color=mycolors[i], label=y)
        plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'value'][-1:].values[0], y, fontsize=12, color=mycolors[i])

# Decoration
plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$Drug Sales$', xlabel='$Month$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot of Drug Sales Time Series", fontsize=20)
plt.show()

## Boxplot of Month-wise (Seasonal) and Year-wise (trend) Distribution
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')
df.reset_index(inplace=True)

# Prepare data
df['year'] = [d.year for d in df.date]
df['month'] = [d.strftime('%b') for d in df.date]
years = df['year'].unique()

# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='value', data=df, ax=axes[0])
sns.boxplot(x='month', y='value', data=df.loc[~df.year.isin([1991, 2008]), :])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()
###############################################################################
## Patterns in a time series
"""
Any time series may be split into the following components: 
    Base Level + Trend + Seasonality + Error
A trend is observed when there is an increasing or decreasing slope observed in 
the time series. Whereas seasonality is observed when there is a distinct 
repeated pattern observed between regular intervals due to seasonal factors. 
It could be because of the month of the year, the day of the month, weekdays or 
even time of the day.

Additive time series:
Value = Base Level + Trend + Seasonality + Error

Multiplicative Time Series:
Value = Base Level x Trend x Seasonality x Error

‘cyclic’ vs ‘seasonal’ pattern:
If the patterns are not of fixed calendar based frequencies, then it is cyclic. 
Because, unlike the seasonality, cyclic effects are typically influenced by the 
business and other socio-economic factors.
"""

fig, axes = plt.subplots(1,3, figsize=(20,4), dpi=100)
pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/guinearice.csv', parse_dates=['date'], index_col='date').plot(title='Trend Only', legend=False, ax=axes[0])

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv', parse_dates=['date'], index_col='date').plot(title='Seasonality Only', legend=False, ax=axes[1])

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/AirPassengers.csv', parse_dates=['date'], index_col='date').plot(title='Trend and Seasonality', legend=False, ax=axes[2])

## Decompose a time series into its components
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# Import Data
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')

# Multiplicative Decomposition 
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()

# Extract the Components. Let’s extract them (the numerical output of the trend, 
# seasonal and residual components are stored in the result_mul output itself) 
# and put it in a dataframe
# Actual Values = Product of (Seasonal * Trend * Resid)
df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']
print(df_reconstructed.head())
################################################################################
### (Non/)Stationary Time Series
"""
A stationary series is one where the values of the series is not a function of time.
That is, the statistical properties of the series like mean, variance and 
autocorrelation are constant over time. Autocorrelation of the series is nothing 
but the correlation of the series with its previous values, more on this coming up.
A stationary time series id devoid of seasonal effects as well.
Make series stationary by:
    - Differencing the Series (once or more) (The most common)
    - Take the log of the series
    - Take the nth root of the series
    - Combination of the above

Why transformation:
Forecasting a stationary series is relatively easy and the forecasts are more 
reliable. An important reason is, autoregressive forecasting models are 
essentially  linear regression models that utilize the lag(s) of the series 
itself as predictors. We know that linear regression works best if the 
predictors (X variables) are not correlated against each other. So,
stationarizing the series solves this problem since it removes any persistent 
autocorrelation, thereby making the predictors(lags of the series) in the 
forecasting models nearly independent.

Test for stationarity:
    - looking at the plot of the series
    - split the series into 2 or more contiguous parts and computing the 
    summary statistics like the mean, variance and the autocorrelation
    - Augmented Dickey Fuller test (ADF Test)
    - Kwiatkowski-Phillips-Schmidt-Shin – KPSS test (trend stationary)
    - Philips Perron test (PP Test)

Difference between white noise and a stationary series?
Like a stationary series, the white noise is also not a function of time, that 
is its mean and variance does not change over time. But the difference is, the 
white noise is completely random with a mean of 0.
"""

from statsmodels.tsa.stattools import adfuller, kpss
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])

# ADF Test
result = adfuller(df.value.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# KPSS Test
result = kpss(df.value.values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
    
# Difference between white noise and a stationary series
randvals = np.random.randn(1000)
pd.Series(randvals).plot(title='Random White Noise', color='k')
###############################################################################
### Detrend and deseasonalize a TS
"""
Why do we Detrend data?
When you detrend data, you remove an aspect from the data that you think is 
causing some kind of distortion. For example, you might detrend data that 
shows an overall increase, in order to see subtrends. Usually, these subtrends 
are seen as fluctuations on a time series graph.

Detrending a time series is to remove the trend component from a time series. 
But how to extract the trend? There are multiple approaches:
    - Subtract the line of best fit from the time series. The line of best fit 
     may be obtained from a linear regression model with the time steps as the 
     predictor. For more complex trends, you may want to use quadratic terms 
     (x^2) in the model.
    - Subtract the trend component obtained from time series decomposition we 
     saw earlier.
    - Subtract the mean
    - Apply a filter like Baxter-King filter(statsmodels.tsa.filters.bkfilter) 
     or the Hodrick-Prescott Filter (statsmodels.tsa.filters.hpfilter) to remove 
     the moving average trend lines or the cyclical components.
     
Deseasonalize a time series:
1. Take a moving average with length as the seasonal window. This will smoothen 
in series in the process.
2. Seasonal difference the series (subtract the value of previous season from 
the current value)
3. Divide the series by the seasonal index obtained from STL decomposition

Test for seasonality:1. plot the series 2. Autocorrelation Function (ACF) plot
"""
## Detrending
# Using scipy: Subtract the line of best fit
from scipy import signal
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
detrended = signal.detrend(df.value.values)
plt.plot(detrended)
plt.title('Drug Sales detrended by subtracting the least squares fit', fontsize=16)

# Using statmodels: Subtracting the Trend Component.
from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')
detrended = df.value.values - result_mul.trend
plt.plot(detrended)
plt.title('Drug Sales detrended by subtracting the trend component', fontsize=16)

## Deseasonalizing
# Subtracting the Trend Component.
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')

# Time Series Decomposition
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')

# Deseasonalize
deseasonalized = df.value.values / result_mul.seasonal

# Plot
plt.plot(deseasonalized)
plt.title('Drug Sales Deseasonalized', fontsize=16)
plt.plot()

## Seasonality test
from pandas.plotting import autocorrelation_plot
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')

# Draw Plot
plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
autocorrelation_plot(df.value.tolist())
################################################################################
### Treat missing values in a time series
"""
In statistics, imputation is the process of replacing missing data with 
substituted values.
Sometimes, your time series will have missing dates/times. That means, the data 
was not captured or was not available for those periods. It could so happen the 
measurement was zero on those days, in which case, case you may fill up those 
periods with zero.
Secondly, when it comes to time series, you should typically NOT replace 
missing values with the mean of the series, especially if the series is not 
stationary. What you could do instead for a quick and dirty workaround is to 
forward-fill the previous value.
However, depending on the nature of the series, you want to try out multiple 
approaches before concluding. Some effective alternatives to imputation are:
    - Backward Fill
    - Linear Interpolation
    - Quadratic interpolation
    - Mean of nearest neighbors
    - Mean of seasonal couterparts 
It is also possible to consider the following approaches depending on how 
accurate the imputations is desired:

    - If you have explanatory variables use a prediction model like the random 
    forest or k-Nearest Neighbors to predict it.
    - If you have enough past observations, forecast the missing values.
    - If you have enough future observations, backcast the missing values
    - Forecast of counterparts from previous cycles.

"""
# # Generate dataset
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
df_orig = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date').head(100)
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10_missings.csv', parse_dates=['date'], index_col='date')

fig, axes = plt.subplots(7, 1, sharex=True, figsize=(10, 12))
plt.rcParams.update({'xtick.bottom' : False})

## 1. Actual -------------------------------
df_orig.plot(title='Actual', ax=axes[0], label='Actual', color='red', style=".-")
df.plot(title='Actual', ax=axes[0], label='Actual', color='green', style=".-")
axes[0].legend(["Missing Data", "Available Data"])

## 2. Forward Fill --------------------------
df_ffill = df.ffill()
error = np.round(mean_squared_error(df_orig['value'], df_ffill['value']), 2)
df_ffill['value'].plot(title='Forward Fill (MSE: ' + str(error) +")", ax=axes[1], label='Forward Fill', style=".-")

## 3. Backward Fill -------------------------
df_bfill = df.bfill()
error = np.round(mean_squared_error(df_orig['value'], df_bfill['value']), 2)
df_bfill['value'].plot(title="Backward Fill (MSE: " + str(error) +")", ax=axes[2], label='Back Fill', color='firebrick', style=".-")

## 4. Linear Interpolation ------------------
df['rownum'] = np.arange(df.shape[0])
df_nona = df.dropna(subset = ['value'])
f = interp1d(df_nona['rownum'], df_nona['value'])
df['linear_fill'] = f(df['rownum'])
error = np.round(mean_squared_error(df_orig['value'], df['linear_fill']), 2)
df['linear_fill'].plot(title="Linear Fill (MSE: " + str(error) +")", ax=axes[3], label='Cubic Fill', color='brown', style=".-")

## 5. Cubic Interpolation --------------------
f2 = interp1d(df_nona['rownum'], df_nona['value'], kind='cubic')
df['cubic_fill'] = f2(df['rownum'])
error = np.round(mean_squared_error(df_orig['value'], df['cubic_fill']), 2)
df['cubic_fill'].plot(title="Cubic Fill (MSE: " + str(error) +")", ax=axes[4], label='Cubic Fill', color='red', style=".-")

# Interpolation References:
# https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
# https://docs.scipy.org/doc/scipy/reference/interpolate.html

## 6. Mean of 'n' Nearest Past Neighbors ------
def knn_mean(ts, n):
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            n_by_2 = np.ceil(n/2)
            lower = np.max([0, int(i-n_by_2)])
            upper = np.min([len(ts)+1, int(i+n_by_2)])
            ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
            out[i] = np.nanmean(ts_near)
    return out

df['knn_mean'] = knn_mean(df.value.values, 8)
error = np.round(mean_squared_error(df_orig['value'], df['knn_mean']), 2)
df['knn_mean'].plot(title="KNN Mean (MSE: " + str(error) +")", ax=axes[5], label='KNN Mean', color='tomato', alpha=0.5, style=".-")

## 7. Seasonal Mean ----------------------------
def seasonal_mean(ts, n, lr=0.7):
    """
    Compute the mean of corresponding seasonal periods
    ts: 1D array-like of the time series
    n: Seasonal window length of the time series
    """
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            ts_seas = ts[i-1::-n]  # previous seasons only
            if np.isnan(np.nanmean(ts_seas)):
                ts_seas = np.concatenate([ts[i-1::-n], ts[i::n]])  # previous and forward
            out[i] = np.nanmean(ts_seas) * lr
    return out

df['seasonal_mean'] = seasonal_mean(df.value, n=12, lr=1.25)
error = np.round(mean_squared_error(df_orig['value'], df['seasonal_mean']), 2)
df['seasonal_mean'].plot(title="Seasonal Mean (MSE: " + str(error) +")", ax=axes[6], label='Seasonal Mean', color='blue', alpha=0.5, style=".-")

################################################################################
### Autocorrelation and partial autocorrelation functions
"""
Autocorrelation is simply the correlation of a series with its own lags. If a 
series is significantly autocorrelated, that means, the previous values of the 
series (lags) may be helpful in predicting the current value.
Partial Autocorrelation also conveys similar information but it conveys the 
pure correlation of a series and its lag, excluding the correlation 
contributions from the intermediate lags.

The partial autocorrelation of lag (k) of a series is the coefficient of that 
lag in the autoregression equation of Y. The autoregressive equation of Y is 
nothing but the linear regression of Y with its own lags as predictors.

A Lag plot is a scatter plot of a time series against a lag of itself. It is 
normally used to check for autocorrelation. If there is any pattern existing in 
the series like the one you see below, the series is autocorrelated. If there 
is no such pattern, the series is likely to be random white noise.
"""

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')

# Calculate ACF and PACF upto 50 lags
# acf_50 = acf(df.value, nlags=50)
# pacf_50 = pacf(df.value, nlags=50)

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df.value.tolist(), lags=50, ax=axes[0])
plot_pacf(df.value.tolist(), lags=50, ax=axes[1])

## lag plot
from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})

# Import
ss = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv')
a10 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')

# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(ss.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plots of Sun Spots Area \n(Points get wide and scattered with increasing lag -> lesser correlation)\n', y=1.15)    

fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(a10.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plots of Drug Sales', y=1.05)    
plt.show()

################################################################################
### Estimate the forecastability of a time series
"""
The more regular and repeatable patterns a time series has, the easier it is to 
forecast. The ‘Approximate Entropy’ can be used to quantify the regularity and 
unpredictability of fluctuations in a time series.
The higher the approximate entropy, the more difficult it is to forecast it.

Another better alternate is the ‘Sample Entropy’.
Sample Entropy is similar to approximate entropy but is more consistent in 
estimating the complexity even for smaller time series. For example, a random 
time series with fewer data points can have a lower ‘approximate entropy’ than 
a more ‘regular’ time series, whereas, a longer random time series will have a 
higher ‘approximate entropy’.
"""

# https://en.wikipedia.org/wiki/Approximate_entropy
ss = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv')
a10 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')
rand_small = np.random.randint(0, 100, size=36)
rand_big = np.random.randint(0, 100, size=136)

def ApEn(U, m, r):
    """Compute Aproximate entropy"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m+1) - _phi(m))

print(ApEn(ss.value, m=2, r=0.2*np.std(ss.value)))     # 0.651
print(ApEn(a10.value, m=2, r=0.2*np.std(a10.value)))   # 0.537
print(ApEn(rand_small, m=2, r=0.2*np.std(rand_small))) # 0.143
print(ApEn(rand_big, m=2, r=0.2*np.std(rand_big)))     # 0.716

# https://en.wikipedia.org/wiki/Sample_entropy
def SampEn(U, m, r):
    """Compute Sample entropy"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m+1) / _phi(m))

print(SampEn(ss.value, m=2, r=0.2*np.std(ss.value)))      # 0.78
print(SampEn(a10.value, m=2, r=0.2*np.std(a10.value)))    # 0.41
print(SampEn(rand_small, m=2, r=0.2*np.std(rand_small)))  # 1.79
print(SampEn(rand_big, m=2, r=0.2*np.std(rand_big)))      # 2.42

###############################################################################
### smoothen a time series: Why and how
"""
Smoothening of a time series may be useful in:
- Reducing the effect of noise in a signal get a fair approximation of the noise-filtered series.
- The smoothed version of series can be used as a feature to explain the original series itself.
- Visualize the underlying trend better

So how to smoothen a series? Let’s discuss the following methods:

    - Take a moving average
    - Do a LOESS smoothing (Localized Regression)
    - Do a LOWESS smoothing (Locally Weighted Regression)

Moving average is nothing but the average of a rolling window of defined width. 
But you must choose the window-width wisely, because, large window-size will 
over-smooth the series. For example, a window-size equal to the seasonal duration 
(ex: 12 for a month-wise series), will effectively nullify the seasonal effect.

LOESS, short for ‘LOcalized regrESSion’ fits multiple regressions in the local 
neighborhood of each point. It is implemented in the statsmodels package, 
where you can control the degree of smoothing using frac argument which specifies 
the percentage of data points nearby that should be considered to fit a 
regression model.
"""
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# Import
df_orig = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')

# 1. Moving Average
df_ma = df_orig.value.rolling(3, center=True, closed='both').mean()

# 2. Loess Smoothing (5% and 15%)
df_loess_5 = pd.DataFrame(lowess(df_orig.value, np.arange(len(df_orig.value)), frac=0.05)[:, 1], index=df_orig.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df_orig.value, np.arange(len(df_orig.value)), frac=0.15)[:, 1], index=df_orig.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df_orig['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()
################################################################################
### Granger Causality: test to know if one TS is helpful in forecasting another
"""
It is based on the idea that if X causes Y, then the forecast of Y based on 
previous values of Y AND the previous values of X should outperform the forecast 
of Y based on previous values of Y alone.
"""
from statsmodels.tsa.stattools import grangercausalitytests
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
df['month'] = df.date.dt.month
grangercausalitytests(df[['value', 'month']], maxlag=2)
###############################################################################
### Frequencies, resampling and rolling windows
"""
https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

Frequencies:
When the data points of a time series are uniformly spaced in time (e.g., 
hourly, daily, monthly, etc.), the time series can be associated with a 
frequency in pandas.
If you’re doing any time series analysis which requires uniformly spaced data 
without any missings, you’ll want to use asfreq() to convert your time series 
to the specified frequency and fill any missings with an appropriate method.

Resampling:
It is often useful to resample our time series data to a lower or higher 
frequency. Resampling to a lower frequency (downsampling) usually involves an 
aggregation operation — for example, computing monthly sales totals from daily 
data. The daily OPSD data we’re working with in this tutorial was downsampled 
from the original hourly time series. Resampling to a higher frequency 
(upsampling) is less common and often involves interpolation or other data 
filling method — for example, interpolating hourly weather data to 10 minute 
intervals for input to a scientific model.
DataFrame’s resample() method=> splits the DatetimeIndex into time bins and 
groups the data by time bin. The resample() method returns a Resampler object, 
similar to a pandas GroupBy object. We can then apply an aggregation method 
such as mean(), median(), sum(), etc., to the data group for each time bin.

Rolling Windows:
Rolling window operations are another important transformation for time series 
data. Similar to downsampling, rolling windows split the data into time windows 
and the data in each window is aggregated with a function such as mean(), 
median(), sum(), etc. However, unlike downsampling, where the time bins do not 
overlap and the output is at a lower frequency than the input, rolling windows 
overlap and “roll” along at the same frequency as the data, so the transformed 
time series is at the same frequency as the original time series.

By default, all data points within a window are equally weighted in the aggregation, but this can be changed by specifying window types such as Gaussian, triangular, and others.
"""
opsd_daily = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv', index_col=0, parse_dates=True)
#opsd_daily.index
# Add columns with year, month, and weekday name
opsd_daily['Year'] = opsd_daily.index.year
opsd_daily['Month'] = opsd_daily.index.month
opsd_daily['Weekday Name'] = opsd_daily.index.weekday_name
# Display a random sampling of 5 rows
print(opsd_daily.sample(5, random_state=0))

# To select an arbitrary sequence of date/time values from a pandas time series,
# we need to use a DatetimeIndex, rather than simply a list of date/time strings
times_sample = pd.to_datetime(['2013-02-03', '2013-02-06', '2013-02-08'])
# Select the specified dates and just the Consumption column
consum_sample = opsd_daily.loc[times_sample, ['Consumption']].copy()
print(consum_sample)

# Convert the data to daily frequency, without filling any missings
consum_freq = consum_sample.asfreq('D')
# Create a column with missings forward filled
consum_freq['Consumption - Forward Fill'] = consum_sample.asfreq('D', method='ffill')
print(consum_freq)

## Resampling
# Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
# Resample to weekly frequency, aggregating with mean
opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean()
opsd_weekly_mean.head(3)

print('opsd_daily', opsd_daily.shape[0])
print('opsd_weekly', opsd_weekly_mean.shape[0])

# Plot the daily and weekly Solar TS together over a single 6-month period to compare them.
# Start and end of the date range to extract
start, end = '2017-01', '2017-06'
# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()
ax.plot(opsd_daily.loc[start:end, 'Solar'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('Solar Production (GWh)')
ax.legend()

# Compute the monthly sums, setting the value to NaN for any month which has
# fewer than 28 days of data
opsd_monthly = opsd_daily[data_columns].resample('M').sum(min_count=28)
opsd_monthly.head(3)

# Explore the monthly TS by plotting the electricity consumption as a line plot, 
# and the wind and solar power production together as a stacked area plot.
import matplotlib.dates as mdates
fig, ax = plt.subplots()
ax.plot(opsd_monthly['Consumption'], color='black', label='Consumption')
opsd_monthly[['Wind', 'Solar']].plot.area(ax=ax, linewidth=0)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.set_ylabel('Monthly Total (GWh)')

# Compute the annual sums, setting the value to NaN for any year which has
# fewer than 360 days of data
opsd_annual = opsd_daily[data_columns].resample('A').sum(min_count=360)
# The default index of the resampled DataFrame is the last day of each year,
# ('2006-12-31', '2007-12-31', etc.) so to make life easier, set the index
# to the year component
opsd_annual = opsd_annual.set_index(opsd_annual.index.year)
opsd_annual.index.name = 'Year'
# Compute the ratio of Wind+Solar to Consumption
opsd_annual['Wind+Solar/Consumption'] = opsd_annual['Wind+Solar'] / opsd_annual['Consumption']
opsd_annual.tail(3)

# Plot from 2012 onwards, because there is no solar production data in earlier years
ax = opsd_annual.loc[2012:, 'Wind+Solar/Consumption'].plot.bar(color='C0')
ax.set_ylabel('Fraction')
ax.set_ylim(0, 0.3)
ax.set_title('Wind + Solar Share of Annual Electricity Consumption')
plt.xticks(rotation=0)

## Rolling window
# Compute the centered 7-day rolling mean
opsd_7d = opsd_daily[data_columns].rolling(7, center=True).mean()
opsd_7d.head(10)

# Start and end of the date range to extract
start, end = '2017-01', '2017-06'
# Plot daily, weekly resampled, and 7-day rolling mean time series together
fig, ax = plt.subplots()
ax.plot(opsd_daily.loc[start:end, 'Solar'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.plot(opsd_7d.loc[start:end, 'Solar'],
marker='.', linestyle='-', label='7-d Rolling Mean')
ax.set_ylabel('Solar Production (GWh)')
ax.legend()
