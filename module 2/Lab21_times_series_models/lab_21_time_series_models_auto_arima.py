# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:17:10 2022

@author: rana_
"""

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns


data = pd.read_csv(r"C:\Users\rana_\IronhackDA\DAFT_212\module_2\Lab_21_Time-Series-Models\airlines_passengers.csv")

data["Month"]= pd.to_datetime(data["Month"])
data
data.set_index('Month', inplace=True)

data

#Because the data is not sationary, we shoul make it stationary
# Change for (t)th day is Close for (t)th day minus Close for (t-1)th day.
data['Difference'] = data['Passengers'].diff()

# Plot the Change
plt.figure(figsize=(10, 7))
plt.plot(data['Difference'])
plt.title('First Order Differenced Series', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Difference', fontsize=12)
plt.show()


data['Date'] = data.index
train = data[data['Date'] < pd.to_datetime("1960-08", format='%Y-%m')]
train['train'] = train['Passengers']
del train['Date']
del train['Passengers']
test = data[data['Date'] >= pd.to_datetime("1960-08", format='%Y-%m')]
del test['Date']
test['test'] = test['Passengers']
del test['Passengers']
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.title("Train/Test split for Passenger Data")
plt.ylabel("Passenger Number")
plt.xlabel('Year-Month')
sns.set()
plt.show()


from pmdarima.arima import auto_arima
model = auto_arima(train["Difference"].dropna(), trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train["Difference"].dropna())
forecast = model.predict()

forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])

from statsmodels.tools.eval_measures import rmse
rmse(train["Difference"].dropna(),model.predict(len(train)-1))

rmse(data["Difference"].dropna(),model.predict(len(data)-1))
from statsmodels.tsa.arima.model import ARIMA
