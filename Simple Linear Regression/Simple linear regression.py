#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing liberies 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Read Data

df = pd.read_csv("delivery_time.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# # Feature Engineering

# In[6]:


# Renaming Columns
df=df.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
df.head()


# # EDA and Data Visualization

# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


sns.distplot(df['delivery_time'])


# In[10]:


sns.distplot(df['sorting_time'])


# # Correlation Analysis

# In[11]:


df.corr()


# In[12]:


sns.heatmap(df.corr(), annot=True)


# In[13]:


sns.regplot(df['delivery_time'], df['sorting_time'])


# # Model Building

# In[14]:


model = smf.ols('delivery_time ~ sorting_time', data=df).fit()
model


# # Model Testing

# In[15]:


# Finding Coefficient parameters
model.params


# In[16]:


# Finding tvalues and pvalues
print(model.tvalues,model.pvalues)


# In[17]:


# Finding Rsquared Values
print(model.rsquared, model.rsquared_adj)


# # Model Predictions

# In[20]:


# Manual prediction 
# y = m(x) + c

delivery_time = (1.649020 * 16)+ 6.582734
delivery_time


# In[21]:


# Automatic Prediction

new_data = pd.Series([2,6,8,9])
new_data = pd.DataFrame(new_data,columns=['sorting_time'])
new_data


# In[22]:


model.predict(new_data)


# In[23]:


df['log_delivery_time'] = np.log(df['delivery_time'])
df['square_delivery_time'] = np.square(df['delivery_time'])
df['sqrt_delivery_time'] = np.sqrt(df['delivery_time'])

df['log_sorting_time'] = np.log(df['sorting_time'])


# In[24]:


df.head()


# # Transforming data

# In[25]:


x = df.iloc[:,0].values
y = df.iloc[:,1].values
x, y


# In[27]:


plt.scatter(x,y)


# In[28]:


re = LinearRegression()
re.fit(x.reshape(-1,1), y)

ypred =  re.predict(x.reshape(-1,1))
ypred

print(r2_score(y,ypred))
print(np.sqrt(mean_squared_error(y, ypred)))


# In[29]:


# Transforming data (sqrt)

ysqrt = np.sqrt(y)
xsqrt = np.sqrt(x.reshape(-1,1))

re = LinearRegression()
re.fit(xsqrt.reshape(-1,1), y)

ypred =  re.predict(xsqrt.reshape(-1,1))

print(r2_score(ysqrt,ypred))
print(np.sqrt(mean_squared_error(ysqrt,ypred)))


# In[30]:


# Transforming data (log)

ylog = np.log(y)
xlog = np.log(x.reshape(-1,1))

re = LinearRegression()
re.fit(xlog.reshape(-1,1), y)

ypred =  re.predict(xlog.reshape(-1,1))

print(r2_score(ylog,ypred))
print(np.sqrt(mean_squared_error(ylog,ypred)))


# In[31]:


# Transforming data (Square)

ysquare = np.square(y)
xsquare = np.square(x.reshape(-1,1))

re = LinearRegression()
re.fit(xsquare.reshape(-1,1), y)

ypred =  re.predict(xsquare.reshape(-1,1))

print(r2_score(ysquare,ypred))
print(np.sqrt(mean_squared_error(ysquare,ypred)))


# # Salary Hike

# In[33]:


# Read data

df = pd.read_csv("Salary_Data.csv")
df.head()


# # EDA and Data Visualization

# In[34]:


df.describe()


# In[35]:


df.info()


# In[37]:


sns.distplot(df['YearsExperience'])


# In[38]:


sns.distplot(df['Salary'])


# # Correlation Analysis

# In[39]:


df.corr()


# In[40]:


sns.heatmap(df.corr(), annot=True)


# In[41]:


sns.regplot(df['Salary'], df['YearsExperience'])


# # Model Building

# In[42]:


model = smf.ols('Salary ~ YearsExperience', data=df).fit()


# # Model Testing

# In[43]:


# Finding Coefficient parameters
model.params


# In[44]:


# Finding tvalues and pvalues
print(model.tvalues,model.pvalues)


# In[45]:


# Finding Rsquared Values
print(model.rsquared, model.rsquared_adj)


# # Model Predictions

# In[46]:


# Manual prediction 
# y = m(x) + c

delivery_time = (9449.962321 * 5) + 25792.200199
delivery_time


# In[47]:


# Automatic Prediction

new_data = pd.Series([2,6,4,8,9])
new_data = pd.DataFrame(new_data,columns=['YearsExperience'])
new_data


# In[48]:


model.predict(new_data)


# # Transforming data

# In[49]:


x = df.iloc[:,0].values
y = df.iloc[:,1].values
x, y


# In[50]:


plt.scatter(x,y)


# In[51]:


re = LinearRegression()
re.fit(x.reshape(-1,1), y)

ypred =  re.predict(x.reshape(-1,1))
ypred

print(r2_score(y,ypred))
print(np.sqrt(mean_squared_error(y, ypred)))


# In[52]:


# Transforming data (sqrt)

ysqrt = np.sqrt(y)
xsqrt = np.sqrt(x.reshape(-1,1))

re = LinearRegression()
re.fit(xsqrt.reshape(-1,1), y)

ypred =  re.predict(xsqrt.reshape(-1,1))

print(r2_score(ysqrt,ypred))
print(np.sqrt(mean_squared_error(ysqrt,ypred)))


# In[ ]:




