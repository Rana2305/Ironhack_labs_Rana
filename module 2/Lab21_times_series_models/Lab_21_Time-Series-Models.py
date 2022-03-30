#!/usr/bin/env python
# coding: utf-8

# We will be working with publicly available airline passenger time series data. To start, let’s import the Pandas library and read the airline passenger data into a data frame

# In[32]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


get_ipython().system(' pip install pmdarima')


# In[33]:


data = pd.read_csv(r"C:\Users\rana_\IronhackDA\DAFT_212\module_2\Lab_21_Time-Series-Models\airlines_passengers.csv")


# Let’s display the first five rows of data

# In[34]:


data.head(5)


# We can see that the data contains a column labeled “Month” that contains dates. In that column, 
# the dates are formatted as year–month. We also see that the data starts in the year 1949.
# The second column is labeled Passengers, and it contains the number of passengers for the year–month. Let’s take a look at the last five records the data 

# In[35]:


data.tail(5)


# In[36]:


data.dtypes


# In[ ]:





# We see that the data ends in 1960. The next thing we will want to do is convert the month column into a datetime object. 
# This will allow it to programmatically pull time values like the year or month for each record.
# To do this, we use the Pandas to_datetime() method. Note that this process automatically inserts the first day of each month, which is basically a dummy value since we have no daily passenger data.

# In[37]:


data["Month"]= pd.to_datetime(data["Month"])
data


# The next thing we can do is convert the month column to an index. 
# 

# In[38]:


data.set_index('Month', inplace=True)

data


# Let’s generate a time series plot using Seaborn and Matplotlib. This will allow us to visualize the time series data. 
# Import the libraries and generate the lineplot. Label the y-axis with Matplotlib “Number of Passengers”

# In[39]:


data.plot()
plt.ylabel('Number od Passengers')
plt.show()


# In[40]:


import seaborn as sns
sns.lineplot(x="Month", y="Passengers", data=data)
plt.xticks(rotation=20)
plt.title('Number of passengers per month')
plt.show()


# Stationarity is a key part of time series analysis. Import the augmented Dickey-Fuller test from the statsmodels package.

# In[41]:


from statsmodels.tsa.stattools import adfuller


# Let’s pass our data frame into the adfuller method. 
# Here, we specify the autolag parameter as “AIC”, which means that the lag is chosen to minimize the information criterion

# In[42]:


result = adfuller(data, autolag='AIC' )
print('ADF Test Statistic: %.2f' % result[0])
print('5%% Critical Value: %.2f' % result[4]['5%'])
print('p-value: %.2f' % result[1])


# Store our results in a dataframe display it

# In[ ]:





# Explain the results

# The p-value is greater than 0.05. We fait to reject the null hypothesis and conclude that the time series is not stationary.

# Autocorrelation
# Checking for autocorrelation in time series data is another important part of the analytic process. 
# This is a measure of how correlated time series data is at a given point in time with past values, 
# which has huge implications across many industries. For example, if our passenger data has strong autocorrelation, we can assume that high passenger numbers today suggest a strong likelihood that they will be high tomorrow as well.
# Please calculate and show the autocorrelation

# In[43]:


import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
# Calculate the ACF (via statsmodel)
plot_acf(data)
# Show the data as a plot (via matplotlib)
plt.show()


# Calculate partial autocorrelation

# In[44]:


from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
plot_pacf(data, alpha =0.05, lags=50)
plt.show()


# In[45]:


statsmodels.tsa.stattools.pacf(data.passengers, nlags=50, alpha=.05)


# Decomposition
# Trend decomposition is another useful way to visualize the trends in time series data.
# To proceed, let’s import seasonal_decompose from the statsmodels package:
# from statsmodels.tsa.seasonal import seasonal_decompose
# 

# In[46]:


from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse


# Next, let’s pass our data frame into the seasonal_decompose method and plot the result:
# decompose = 
# decompose.plot()
# plt.show()
# 

# In[47]:


# Import Dataon 
result_add = seasonal_decompose(data['Passengers'], model='additive') 


# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_add.plot().suptitle('additive Decompose', fontsize=22)

plt.show()


# In[48]:


result_mul = seasonal_decompose(data['Passengers'], model='multiplicative') #just the colums and the model mutiplicative


# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('multiplicative Decompose', fontsize=22)

plt.show()


# 
# Can you describe the charts?
# 

# Seeing this image we can very clearly observe variance of seasonality and residual component is constant for multiplicative decompose. So the time series is a multiplicative time series.

# Let's check the models on our dataset. Please note, if the dataset is non-stationary, you need to make it stationary

# In[49]:


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


# In[50]:


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


# In[51]:


from pmdarima.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])


# MA(1)

# In[58]:


from statsmodels.tsa.arima.model import ARIMA


# In[57]:



model = ARIMA(endog=data['Passengers'], order=(1, 1, 1))

results1 = model.fit()
results1.summary()



# Display the output

# In[53]:


plt.plot(results1.fittedvalues, color='red')


# In[54]:


data


# MA(2)

# In[55]:


model = ARIMA(endog=data['Passengers'], order=(1, 1, 2))

results2 = model.fit()
results2=model.predict(len(data))
results2.summary()


# Display the output

# In[56]:


plt.plot(results2.fittedvalues, color='red')


# AR(2)

# In[57]:


model = ARIMA(endog=data['Passengers'], order=(2, 1, 1))

results3 = model.fit()
results3.summary()


# ARMA (?,?)

# In[58]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(endog=data['Passengers'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

results4 = model.fit(disp=False)
results4.summary()


# Display the output

# In[59]:


plt.plot(results4.fittedvalues, color='red')


# ARIMA(?,?)

# In[60]:


model = SARIMAX(endog=data['Passengers'], order=(1, 1, 2), seasonal_order=(1, 1, 1, 6))

results5 = model.fit(disp=False)
results5.summary()


# Display the output

# In[61]:


plt.plot(results5.fittedvalues, color='red')


# In[67]:


a=rmse(data['Passengers'],results1.fittedvalues)
a


# Let’s calculate root mean squared error (RMSE) for all the models. Explain the values

# In[31]:


from statsmodels.tools.eval_measures import rmse
results=[results1.fittedvalues, results2.fittedvalues, results3.fittedvalues, results4.fittedvalues, results5.fittedvalues]


# In[30]:


for i in results:
    print(i)


# Calculate AIC

# In[ ]:




