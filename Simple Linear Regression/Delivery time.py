#!/usr/bin/env python
# coding: utf-8

# # # Delivery time

# In[1]:


# Importing liberies 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read Data

df = pd.read_csv("delivery_time.csv")
df.head()


# In[3]:


# Renaming Columns
df=df.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
df.head()


# EDA and Data Visualization

# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


sns.distplot(df['delivery_time'])


# In[7]:


sns.distplot(df['delivery_time'])


# Correlation Analysis

# In[8]:


df.corr()


# In[9]:


sns.heatmap(df.corr(), annot=True)


# In[10]:


sns.regplot(df['delivery_time'], df['sorting_time'])


# Model Bulding

# In[11]:


model = smf.ols('delivery_time ~ sorting_time', data=df).fit()
model


# Model Testing

# In[12]:


# Finding Coefficient parameters
model.params


# In[13]:


# Finding tvalues and pvalues
print(model.tvalues,model.pvalues)


# In[14]:


# Finding Rsquared Values
print(model.rsquared, model.rsquared_adj)


# Model Predictions

# In[15]:


# Manual prediction 
# y = m(x) + c

delivery_time = (1.649020 * 16)+ 6.582734
delivery_time


# In[16]:


# Automatic Prediction

new_data = pd.Series([2,6,8,9])
new_data = pd.DataFrame(new_data,columns=['sorting_time'])
new_data


# In[17]:


model.predict(new_data)

