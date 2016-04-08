
# coding: utf-8

# In[1]:

# this script is used to generate preprocessing data for the feature selection and training stage
# it will write several files : df_data.csv, resource_data.csv , donation_data.csv, essay_data.csv
# these files contains extracted features use for training 
# Please run this script before run the date_mine_train.py 
# by Hang Shang  

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

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier,     RandomForestRegressor, RandomForestClassifier
from datetime import date




# start to calculate history rate for different attributes 
id_columns = ['teacher_acctid','schoolid']
location_columns = ['school_city','school_state','school_district','school_county','school_zip']
all_columns = id_columns + location_columns

requirements_columns = ['projectid','date_posted','is_exciting','at_least_1_teacher_referred_donor','fully_funded','at_least_1_green_donation','great_chat','one_or_more_required']
require_only_columns = ['is_exciting','at_least_1_teacher_referred_donor','fully_funded','at_least_1_green_donation','great_chat','one_or_more_required']




def calculate_possibility(df, title, columns=require_only_columns):
    df_p = df 
    alpha = 10
    for var in require_only_columns:
        mean = df_p[var].sum()/df_p['count'].sum()
        #print(mean)
        df_p[var] = (df_p[var]+mean*alpha)/(df_p['count']+alpha)
        df_p[var][df_p['count']==1] = 0      
        df_p.columns = df_p.columns.str.replace(var,"".join(title+'_'+var))
    df_p.rename(columns={'count': "".join(title+'_count')}, inplace=True)
    return df_p
def calculate_possibility_by_time(df,title,start_date,end_date,columns= require_only_columns):
    df_p = df[(df['date_posted']<end_date)&(df['date_posted']>=start_date)]
    alpha = 10
    for var in require_only_columns:
        mean = df_p[var].sum()/df_p['count'].sum()
        #print(mean)
        df_p[var] = (df_p[var]+mean*alpha)/(df_p['count']+alpha)
        df_p[var][df_p['count']==1] = 0      
        df_p.columns = df_p.columns.str.replace(var,"".join(title+'_'+var))
    df_p.rename(columns={'count': "".join(title+'_count')}, inplace=True)
    return df_p




def calculate_sum(df,name):
    #df_tmp = df[df['date_posted']<"2014-01-01"]
    df_tmp = df 
    name_group = df_tmp.groupby(name,as_index=False)
    count = pd.DataFrame({'count':name_group.size()}).reset_index()
    prev = name_group.sum()
    prev = pd.merge(prev,count,on=name)
    df_tmp = pd.merge(df_tmp[['projectid',name]],prev,how='left',on=name)
    return df_tmp

def calculate_by_time(df,name,start_date,end_date):
    df_tmp = df[(df['date_posted']<end_date)&(df['date_posted']>=start_date)]
    name_group = df_tmp.groupby(name,as_index=False)
    count = pd.DataFrame({'count':name_group.size()}).reset_index()
    prev = name_group.sum()
    prev = pd.merge(prev,count,on=name)
    df_tmp = pd.merge(df_tmp[['projectid','date_posted',name]],prev,how='left',on=name)
    df_tmp = df_tmp.drop('date_posted',axis=1)
    return df_tmp

def get_full_df():
    test_df = df
    #data = df[(df['cat']=='train')|(df['cat']=='val')]
    for var in all_columns:
        #get columns of requirements
        name_tmp = df[np.append(requirements_columns,var)]
        # calculate sum accroding to different attribute
        sum_tmp = calculate_sum(name_tmp,var)
        #calculate the possibility of different attribute
        pos_tmp = calculate_possibility(sum_tmp,var)
        # merge the result to the main df dataframe
        test_df = pd.merge(test_df,pos_tmp,how='left',on=['projectid',var])
        print("finish merge for "+ var)
    return test_df




# this function is not finished yet
def get_date_partition(start_date, end_date, interval):
    intervals = []
    start_time = pd.to_datetime(start_date)
    end_time = pd.to_datetime(end_date)
    bet = end_time - start_time
    times = bet/interval
    for i in range(0,times):
        end = start_time + interval
        start_time.dt.strftime()




print('loading the data...')
projects = pd.read_csv('./data/projects.csv')
outcomes = pd.read_csv('./data/outcomes.csv')
#donations = pd.read_csv('./data/donations.csv')
#resources = pd.read_csv('./data/resources.csv')
print('complete..')




print(outcomes.shape)
outcomes = outcomes.sort_values(by='projectid')
outcomes.fillna(method='pad')
outcomes.head()




projects = projects.sort_values(by='projectid')
projects.fillna(method='pad')
 
df = pd.merge(projects, outcomes, how='left', on='projectid')
print(df.shape)
df[df['is_exciting'].isnull()].shape




#add category for the data 
df['cat'] = "train"
df['cat'][df["date_posted"]<"2010-04-01"] = "nouse"
# valadation set
df['cat'][df["date_posted"]>="2013-01-01"] = "val"
# test set
df['cat'][df["date_posted"]>="2014-01-01"]= "test"
df = df[df['cat']!="nouse"]
print(df.shape)




# transform the t / f to 1 and 0
df['is_exciting'][df['is_exciting']=="t"] = 1
df['is_exciting'][df['is_exciting']=="f"] = 0
df['is_exciting'].fillna(0,inplace=True)




df["at_least_1_teacher_referred_donor"][df["at_least_1_teacher_referred_donor"]=="t"] = 1
df["at_least_1_teacher_referred_donor"][df["at_least_1_teacher_referred_donor"]=="f"] = 0
df["at_least_1_teacher_referred_donor"].fillna(0,inplace=True)




df["great_chat"][df["great_chat"]=="t"] = 1
df["great_chat"][df["great_chat"]=="f"] = 0
df["great_chat"].fillna(0,inplace=True)




df["fully_funded"][df["fully_funded"]=="t"] = 1
df["fully_funded"][df["fully_funded"]=="f"] = 0
df["fully_funded"].fillna(0,inplace=True)




df["at_least_1_green_donation"][df["at_least_1_green_donation"]=="t"] = 1
df["at_least_1_green_donation"][df["at_least_1_green_donation"]=="f"] = 0
df["at_least_1_green_donation"].fillna(0,inplace=True)




df["donation_from_thoughtful_donor"][df["donation_from_thoughtful_donor"]=="t"] = 1
df["donation_from_thoughtful_donor"][df["donation_from_thoughtful_donor"]=="f"] = 0
df["donation_from_thoughtful_donor"].fillna(0,inplace=True)




df["three_or_more_non_teacher_referred_donors"][df["three_or_more_non_teacher_referred_donors"]=="t"] = 1
df["three_or_more_non_teacher_referred_donors"][df["three_or_more_non_teacher_referred_donors"]=="f"] = 0
df["three_or_more_non_teacher_referred_donors"].fillna(0,inplace=True)




df["one_non_teacher_referred_donor_giving_100_plus"][df["one_non_teacher_referred_donor_giving_100_plus"]=="t"] = 1
df["one_non_teacher_referred_donor_giving_100_plus"][df["one_non_teacher_referred_donor_giving_100_plus"]=="f"] = 0
df["one_non_teacher_referred_donor_giving_100_plus"].fillna(0,inplace=True)




df['teacher_referred_count'][df['teacher_referred_count']<1] = 0
df['teacher_referred_count'][df['teacher_referred_count']>=1] = 1
df['teacher_referred_count'].fillna(0,inplace=True)




df['non_teacher_referred_count'][df['non_teacher_referred_count']<1] = 0
df['non_teacher_referred_count'][df['non_teacher_referred_count']>=3] = 1
df['non_teacher_referred_count'].fillna(0,inplace=True)




one_or_more_required = ['three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus',
                       'donation_from_thoughtful_donor']
df['one_or_more_required'] = df[one_or_more_required].sum(axis=1)
df['one_or_more_required'][df['one_or_more_required']>=1] = 1
df['one_or_more_required'][df['one_or_more_required']<1] = 0
# this was indicated in the great_chat no need to cal again 
#df['great_messages_proportion'].fillna(0,inplace=True)
#df['great_messages_proportion'][df['great_messages_proportion'] >= 62] = True
#df['great_messages_proportion'][df['great_messages_proportion'] < 62] = False
#df['great_messages_proportion'].apply(lambda x: 1 if x else 0)




# add time tag columns for the data, maybe used for calculate history features
df["year"] = df["date_posted"].apply(lambda x: x.split("-")[0])
df["month"] = df["date_posted"].apply(lambda x: x.split("-")[1])
df["day"] = df["date_posted"].apply(lambda x: x.split("-")[2])
df['school_ncesid'] = df['school_ncesid'].apply(str)
df['school_zip'] = df['school_zip'].apply(str)



df['grade_level'][df['grade_level']=="Grades PreK-2"] = 0.0
df['grade_level'][df['grade_level']=="Grades 3-5"] = 1.0
df['grade_level'][df['grade_level']=="Grades 6-8"] = 2.0
df['grade_level'][df['grade_level']=="Grades 9-12"] = 3.0




df['poverty_level'][df['poverty_level']=='highest poverty'] = 3.0
df['poverty_level'][df['poverty_level']=='high poverty'] = 2.0
df['poverty_level'][df['poverty_level']=='moderate poverty'] = 1.0
df['poverty_level'][df['poverty_level']=='low poverty'] = 0.0



origin_df = df
example = get_full_df()
example.head()





example.to_csv('df_data.csv')




df.head()



donations = pd.read_csv('./data/donations.csv')
print(donations.shape)




donations['is_teacher_acct'][donations["is_teacher_acct"]=="t"] = 1.0
donations['is_teacher_acct'][donations["is_teacher_acct"]=="f"] = 0.0




donations = donations[['projectid','is_teacher_acct','donation_total']]
donations['is_teacher_acct'] = donations['is_teacher_acct'].astype(float)
donations_sum = donations.groupby(by='projectid',as_index=False).sum()




donate = pd.merge(df[['projectid','teacher_acctid','schoolid']],donations_sum,how='left',on='projectid')
print(donate.shape)
print(df.shape)
donate.head()




donation_df = donate
print(donation_df.shape)
donor_sum = calculate_sum(donation_df,'teacher_acctid')
donor_sum['is_teacher_acct'] = donor_sum['is_teacher_acct']/donor_sum['count']
donor_sum['donation_total'] = donor_sum['donation_total']/donor_sum['count']
donor_sum['is_teacher_acct'][donor_sum['count']==1] = 0.0
donor_sum['donation_total'][donor_sum['count']==1] = 0.0
donor_sum.rename(columns={'count': "".join('teacher_acctid'+'_count_donor')}, inplace=True)
donor_sum.rename(columns={'is_teacher_acct': "".join('teacher_acctid'+'_is_teacher_acct_donor')}, inplace=True)
donor_sum.rename(columns={'donation_total': "".join('teacher_acctid_donor'+'_donation_total_donor')}, inplace=True)
donor_sum.drop('teacher_acctid',axis=1,inplace=True)
donor_sum.head()




ids = df[['projectid','teacher_acctid','schoolid']]
donate_data = pd.merge(ids,donor_sum,how='left',on='projectid')




print(donate_data.shape)
print(donor_sum.shape)
donate_data.head()




donor_sum_2 = calculate_sum(donation_df,'schoolid')
donor_sum_2['is_teacher_acct'] = donor_sum_2['is_teacher_acct']/donor_sum_2['count']
donor_sum_2['donation_total'] = donor_sum_2['donation_total']/donor_sum_2['count']
donor_sum_2['is_teacher_acct'][donor_sum_2['count']==1] = 0.0
donor_sum_2['donation_total'][donor_sum_2['count']==1] = 0.0
donor_sum_2.rename(columns={'count': "".join('schoolid'+'_count_donor')}, inplace=True)
donor_sum_2.rename(columns={'is_teacher_acct': "".join('schoolid'+'_is_teacher_acct_donor')}, inplace=True)
donor_sum_2.rename(columns={'donation_total': "".join('schoolid'+'_donation_total_donor')}, inplace=True)
donor_sum_2.drop('schoolid',axis=1,inplace=True)
donate_data = pd.merge(donate_data,donor_sum_2,how='left',on='projectid')


donate_data.fillna(0,inplace=True)
donate_data.head()

donate_data.to_csv('donation_data.csv')




resources = pd.read_csv('./data/resources.csv')
print(resources.shape)
resources['cost'] = resources['item_unit_price']*resources['item_quantity']
#resources.head()
total_sum = resources[['projectid','cost','item_quantity']].groupby('projectid',as_index=False).aggregate(np.sum)
total_sum['avg_cost'] = total_sum['cost']/total_sum['item_quantity']
resource_df = df[['projectid','students_reached']]
resource_df = pd.merge(resource_df, total_sum, how='left', on='projectid')
print(df.shape)


# In[38]:

for var in resources['project_resource_type'].unique()[0:6]:
    tmp = resources[(resources['project_resource_type']==var)]
    tmp_sum= tmp[['projectid','cost']].groupby('projectid',as_index=False).aggregate(np.sum)
    tmp_sum.rename(columns={'cost': "".join(var+'_cost')}, inplace=True)
    resource_df = pd.merge(resource_df,tmp_sum,how='left',on ='projectid')
    resource_df["".join(var+'_cost')].fillna(0,inplace=True)
print(resource_df.shape)


# In[39]:

resource_df['price_per_student'] = resource_df['cost']/resource_df['students_reached']
resource_df.to_csv('resource_data.csv')


# In[40]:

essays = pd.read_csv('./data/essays.csv')
essays.fillna("",inplace=True)
essays['essay_length'] = essays['essay'].str.len()
essays['title_length'] = essays['title'].str.len()
essay_data = pd.merge(ids,essays[['projectid','essay_length','title_length']],on='projectid',how='left')


# In[41]:

essay_data.fillna(method='pad',inplace=True)
essay_data.to_csv('essay_data.csv')


# end of script , 