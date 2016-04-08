€
# coding: utf-8

# In[1]:

# imports
import itertools
import os
import re
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn import metrics
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, \
    RandomForestRegressor, RandomForestClassifier
from datetime import date

get_ipython().magic('matplotlib inline')

# In[2]:

print('loading the data...')
projects = pd.read_csv('./data/projects.csv')
outcomes = pd.read_csv('./data/outcomes.csv')
donations = pd.read_csv('./data/donations.csv')
resources = pd.read_csv('./data/resources.csv')
print('complete..')


# In[3]:

print (projects.shape, outcomes.shape)
projects.head()


# In[4]:

outcomes.head()


# In[5]:

# sort the data based on id
projects = projects.sort_values('projectid')
sample = sample.sort_values('projectid')
outcomes = outcomes.sort_values('projectid')


# In[6]:

projects.head()


# In[7]:

totalCount = projects.shape[0]
for i in range(1,projects.shape[1]):
    nullcount = projects[projects[projects.columns[i]].isnull()].shape[0]
    percentage=float(nullcount)/float(totalCount) *100
    if(percentage>0):print(projects.columns[i],percentage,'%')


# In[8]:

projects = projects.fillna(method='pad')


# In[9]:

projects.head(10)


# In[10]:

dates = np.array(projects.date_posted)


# In[11]:

print(dates)


# In[12]:

train_idx = np.where(dates<'2014-01-01')[0]
test_idx = np.where(dates>='2014-01-01')[0]


# In[13]:

print(train_idx)


# In[14]:

print(test_idx)


# In[15]:

projects_numeric_columns = ['school_latitude','school_longitude','fullfullment_labor_materials'
                           'total_price_excluding_optional_support'
                           'total_price_including_optional_support']


# In[21]:

projects_id_columns =['projectid','teacher_acctid','schoolid','school_ncesid']


# In[22]:

projects_categorial_columns = np.array(list(set(projects.columns).difference(set(projects_numeric_columns).union(set(projects_id_columns)))))


# In[23]:

print(projects_categorial_columns)


# In[24]:

projects_categorial_values = np.array(projects[projects_categorial_columns])


# In[25]:

projects[projects_categorial_columns].head()


# In[28]:

# encode the category value and reform the original data
label_encoder = LabelEncoder()
# set up the encoding model, using the first row of data
projects_data = label_encoder.fit_transform(projects_categorial_values[:,0])


# In[30]:

for i in range(1, projects_categorial_values.shape[1]):
    label_encoder = LabelEncoder()
    projects_data = np.column_stack((projects_data, label_encoder.fit_transform(projects_categorial_values[:,i])))

# In[36]:

projects_data = projects_data.astype(float)
print(projects_data)


# In[ ]:



