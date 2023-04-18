#!/usr/bin/env python
# coding: utf-8

# # Salary_Hike

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


# In[3]:


# Read data

df = pd.read_csv("Salary_Data.csv")
df.head()


# # EDA and Data Visualization

# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


sns.distplot(df['YearsExperience'])


# In[7]:


sns.distplot(df['Salary'])


# Correlation Analysis

# In[8]:


df.corr()


# In[9]:


sns.heatmap(df.corr(), annot=True)


# In[10]:


sns.regplot(df['Salary'], df['YearsExperience'])


# # Model Building

# In[11]:


model = smf.ols('Salary ~ YearsExperience', data=df).fit()


# # Model Testing

# In[13]:


# Finding Coefficient parameters
model.params


# In[14]:


# Finding tvalues and pvalues
print(model.tvalues,model.pvalues)


# In[15]:


# Finding Rsquared Values
print(model.rsquared, model.rsquared_adj)


# # Model Predictions

# In[16]:


# Manual prediction 
# y = m(x) + c

delivery_time = (9449.962321 * 5)+ 25792.200199
delivery_time


# In[17]:


# Automatic Prediction

new_data = pd.Series([2,6,4,8,9])
new_data = pd.DataFrame(new_data,columns=['YearsExperience'])
new_data


# In[18]:


model.predict(new_data)

