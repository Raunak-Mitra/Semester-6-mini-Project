
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pmdarima.arima import auto_arima
import streamlit as st
import yfinance as yf

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


from matplotlib.dates import DateFormatter

start = '2010-01-01'
end = '2019-12-31'

st.title("Stock Trend Prediction (Arima excluding Covid Data)")


user_input = st.text_input('Enter Stock Ticker')
# Define the ticker symbol
# ticker_symbol = 'AAPL'  # Example: Apple Inc.

# Fetch historical stock price data using yfinance
stock_data = yf.download(user_input, start='2010-01-01', end='2019-12-31')

#describing data

st.subheader('Data from 2010 - 2019')
st.write(stock_data.describe())

#visualizations

st.subheader("Closing Price vs Time chart")
# fig = plt.figure(figsize =(12,6))
# plt.plot(df.Close)
# st.pyplot(fig)
fig = plt.figure(figsize =(12,6))
plt.plot(stock_data.Close)
st.pyplot(fig)

data_training = pd.DataFrame(stock_data['Close'][0:int(len(stock_data)*0.70)])
data_testing = pd.DataFrame(stock_data['Close'][int(len(stock_data)*0.70):int(len(stock_data))])

# print(data_training.shape)
# print(data_testing.shape)


model_autoARIMA = auto_arima(data_training, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
# print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15,8))
# plt.show()

model = sm.tsa.arima.ARIMA(data_training, order=(1,1,2))
result = model.fit()

# print(result.summary())


pred = result.forecast(739, alpha=0.05)  # 95% conf

pred = pred.to_frame()

data_testing['predicted_mean'] = pred['predicted_mean'].values


st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
# plt.figure(figsize=(10,5), dpi=100)
# plt.plot(data_training, label='training data')
plt.plot(data_testing["Close"], color = 'blue', label='Actual Stock Price')
plt.plot(data_testing["predicted_mean"], color = 'orange',label='Predicted Stock Price')
# plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
plt.title(user_input)
plt.xlabel('Time')
plt.ylabel(user_input)
plt.legend(loc='upper left', fontsize=8)
# plt.show()
st.pyplot(fig2)

mae = np.mean(np.abs(data_testing["predicted_mean"] - data_testing["Close"]))

# Calculate the range of the target variable
target_range = np.max(data_testing["Close"]) - np.min(data_testing["Close"])

# Calculate the error percentage
error_percentage = (mae / target_range) * 100

# error_percentage = (mae / np.mean(data_testing["Close"])) * 100

# Streamlit app
st.subheader("Error Percentage Calculator")

# Display the error percentage
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"Error Percentage: {error_percentage:.2f}%")