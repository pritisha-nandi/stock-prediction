import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import streamlit as st

from datetime import date
from dateutil.relativedelta import relativedelta
start = date.today() - relativedelta(years=5)
end = date.today()

st.title('Stock Prediction')

user_input = st.text_input('Enter Stock Ticker', 'SBIN.NS')
df = pdr.DataReader(user_input, 'yahoo', start, end)

#Describing data
st.subheader('5 year stock data')
st.write(df.describe())

#visulization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(16,8))
plt.plot(df['Close'][0:int(len(df)*0.80)],'b',label = 'Training_data')
plt.plot(df['Close'][int(len(df)*0.80):int(len(df))],'g', label = 'Testing_data')
plt.legend()
st.pyplot(fig)

#LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

model = load_model('lstm_model')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

y_prediction = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_prediction = y_prediction*scale_factor
y_test = y_test*scale_factor

st.subheader('LSTM prediction')
fig2 = plt.figure(figsize = (16,8))
plt.plot(y_test, 'g', label = 'Original Price')
plt.plot(y_prediction, 'r', label = 'Predicted Price')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima

train_data = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
test_data = pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])

model = auto_arima(train_data, start_p=0, start_q=0,
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

import warnings
warnings.filterwarnings('ignore')

model = ARIMA(train_data, order=(1,1,2))  
fitted = model.fit(disp=-1)  
fitted.summary()

fc, se, conf = fitted.forecast(test_data.shape[0], alpha=0.05)

fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
# Plot
st.subheader('ARIMA prediction')
fig3 = plt.figure(figsize=(16,8), dpi=100)
plt.plot(train_data, label='training data')
plt.plot(test_data, 'b', label = 'Original Price')
plt.plot(fc_series, 'r', label = 'Predicted Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('Stock Price Prediction using Arima')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
st.pyplot(fig3)

#comparison
st.subheader('Comparison Between LSTM and ARIMA')

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
#lstmdata
mse1 = mean_squared_error(y_test, y_prediction)
mae1 = mean_absolute_error(y_test, y_prediction)
rmse1 = math.sqrt(mean_squared_error(y_test, y_prediction))
mape1 = np.mean(np.abs(y_test - y_prediction )/np.abs(y_prediction))


#arima data
test = np.array(test_data['Close'])

mse2 = mean_squared_error(test, fc)
mae2 = mean_absolute_error(test, fc)
rmse2 = math.sqrt(mean_squared_error(test, fc))
mape2 = np.mean(np.abs(fc - test)/np.abs(test))

#table
d = {'Model': ['LSTM', 'ARIMA'], 'MSE': [mse1, mse2], 'MAE': [mae1, mae2], 'RMSE': [rmse1, rmse2], 'MAPE': [mape1, mape2]}
d = pd.DataFrame(data=d)
st.table(data=d)