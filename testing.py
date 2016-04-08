
# coding: utf-8

# In[1]:

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

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier,     RandomForestRegressor, RandomForestClassifier
from datetime import date


# In[2]:

print('loading the data...')
df = pd.read_csv('df_data.csv')
donation = pd.read_csv('donation_data.csv')
resource = pd.read_csv('resource_data.csv')
essay = pd.read_csv('essay_data.csv')
print('complete..')

# fisrt partation data into several parts and train and generate based on those pieses
#part_1 = example[(example['date_posted']<"2011-01-01")&(example['date_posted']>="2010-01-01")]
#part_2 = example[(example['date_posted']<"2012-01-01")&(example['date_posted']>="2011-01-01")]
#part_3 = example[(example['date_posted']<"2013-01-01")&(example['date_posted']>="2012-01-01")]
#part_4 = example[(example['date_posted']<"2014-01-01")&(example['date_posted']>="2013-01-01")]


# In[3]:

print(df.shape)
print(donation.shape)
print(resource.shape)
print(essay.shape)


# In[5]:


df.drop('Unnamed: 0',axis=1,inplace=True)
donation.drop('Unnamed: 0',axis=1,inplace=True)
donation.drop('schoolid',axis=1,inplace=True)
donation.drop('teacher_acctid',axis=1,inplace=True)
resource.drop('Unnamed: 0',axis=1,inplace=True)
essay.drop('Unnamed: 0',axis=1,inplace=True)
print(df.columns)
print(donation.columns)
print(resource.columns)
print(essay.columns)
df.head()


# In[6]:


essay.drop('schoolid',axis=1,inplace=True)
essay.drop('teacher_acctid',axis=1,inplace=True)
data = df
data = pd.merge(df,donation,how='left',on='projectid')
print(data.shape)


# In[8]:

resource.drop('students_reached',axis=1,inplace=True)
data = pd.merge(data,resource,how='left',on='projectid')
data = pd.merge(data,essay,how='left',on='projectid')


# In[9]:

data['date_posted'] = pd.to_datetime(data['date_posted'])
ref_date = "2010-01-01"
ref_date = pd.to_datetime(ref_date)
print(ref_date)
data['daysbet'] = data['date_posted'] - ref_date
data['daysbet'] = data['daysbet'].dt.days
data['monthbet'] = data['date_posted'] - ref_date
data['monthbet'] = data['monthbet'].dt.days
data['monthbet'] = data['monthbet']/30


# In[10]:

print(data.shape)
data.head()



# start to calculate history rate for different attributes 
id_columns = ['teacher_acctid','schoolid']
location_columns = ['school_city','school_state','school_district','school_county','school_zip']
all_columns = id_columns + location_columns
one_or_more_required = ['three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus',
                       'donation_from_thoughtful_donor']
requirements_columns = ['projectid','date_posted','is_exciting','at_least_1_teacher_referred_donor','fully_funded','at_least_1_green_donation','great_chat','one_or_more_required']
require_only_columns = ['is_exciting','at_least_1_teacher_referred_donor','fully_funded','at_least_1_green_donation','great_chat','one_or_more_required']

date_columns = ['day','month','year']

projects = pd.read_csv('./data/projects.csv')
sub_primary_requirements = ['great_messages_proportion','teacher_referred_count','non_teacher_referred_count'] 
projects_numeric_columns = ['school_latitude','school_longitude','fulfillment_labor_materials',
                           'total_price_excluding_optional_support',
                           'total_price_including_optional_support','grade_level','poverty_level','students_reached']
projects_id_columns =['projectid','teacher_acctid','schoolid','school_ncesid']
projects_categorial_columns = projects.columns - projects_numeric_columns - projects_id_columns - ['date_posted']

feature_columns = data.columns - data.columns[0:4] - requirements_columns-['cat'] - one_or_more_required-sub_primary_requirements


# In[13]:

data.fillna(method='pad',inplace=True)


# In[14]:

for var in projects_categorial_columns:
    le = LabelEncoder()
    data[var] = le.fit_transform(data[var])
data.tail()


# In[15]:

# load data
train = data[(data['cat']=='train')|(data['cat']=='val')]
feature_data = train[feature_columns]
X = feature_data
names = feature_data.columns.values
Y = train['is_exciting']

test = data[(data['cat']=='test')]
X_test = test[feature_columns]
Y_test = test['is_exciting']


# In[ ]:

# after get full df, do feature selection 
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

rfr = RandomForestClassifier(n_estimators=300,n_jobs=-1,)
print("start to fit the data")
y_ = Y==1
rfr.fit(X,y_)
print("finish fitting")
print( "Features sorted by their score:")
feature_list = sorted(zip(map(lambda x: round(x, 4), rfr.feature_importances_), names), 
             reverse=True)
print(feature_list)


# In[43]:

rfr_predict = rfr.predict_proba(X_test)
rfr_predict = rfr_predict[:,1]
sample = pd.read_csv('./data/sampleSubmission.csv')
sample = sample.sort_values(by='projectid')
sample['is_exciting'] = rfr_predict
sample.to_csv('predictions_rf_2.csv', index = False)


# In[44]:

rfr_predict


# In[41]:
"""
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
dctr = DecisionTreeClassifier()

print("start to fit the data")
dctr.fit(X,Y)
print("finish fitting")
dctr_predict = dctr.predict_proba(X_test)
dctr_predict = dctr_predict[:,1]
print( "Features sorted by their score:")
feature_list_2 = sorted(zip(map(lambda x: round(x, 4), dctr.feature_importances_), names), 
             reverse=True)
print(feature_list_2)
sample = pd.read_csv('./data/sampleSubmission.csv')
sample = sample.sort_values(by='projectid')
sample['is_exciting'] = dctr_predict
sample.to_csv('predictions_dt.csv', index = False)

"""
# In[45]:

# after feature selection, select to drop some columns
update_feature_columns = []
for (v,n) in feature_list:
    update_feature_columns.append(n)

#update_feature_columns_2 = []
#for (v,n) in feature_list_2:
#   update_feature_columns_2.append(n)
#used_feature = list(set(update_feature_columns[0:40]).union(set(update_feature_columns_2[0:40])))
used_feature = list(set(update_feature_columns[0:60]))

# In[46]:

# load data again
train = data[(data['cat']=='train')|(data['cat']=='val')]
feature_data = train[used_feature]
X = feature_data
names = feature_data.columns.values
Y = train['is_exciting']

test = data[(data['cat']=='test')]
X_test = test[used_feature]
Y_test = test['is_exciting']


# In[47]:

#from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
#adb = AdaBoostClassifier(n_estimators=100,learning_rate=0.1)
#adb.fit(X,Y)
#predict = adb.predict(X_test)


# In[ ]:

from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor

# Fit regression model
params = {'n_estimators': 200, 'max_depth': 7, 'min_samples_split': 1,
          'learning_rate': 0.1, 'loss': 'deviance','verbose':1,}
clf = GradientBoostingClassifier(**params)
print ("start training")
clf.fit(X, Y)
print("finish training")


# In[134]:

#predict = adb.predict(X_test)
predict = clf.predict_proba(X_test)
predict = predict[:,1]


# In[135]:

print(predict[predict>0.5].shape)
print(predict[predict<0.5].shape)
predict


# In[ ]:




# In[136]:

sample = pd.read_csv('./data/sampleSubmission.csv')
sample = sample.sort_values(by='projectid')
sample['is_exciting'] = predict
sample.head()


# In[137]:

sample.to_csv('predictions_5_e200_f60.csv', index = False)


# In[ ]:



