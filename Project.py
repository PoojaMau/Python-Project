#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Libraries, Reading Original Data and its Description
# 

# In[2]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# In[3]:


dt=pd.read_excel(r"C:\Users\Pooja Nimbalkar\Desktop\Data Analysis\Python Projects\Project1\E Commerce Dataset.xlsx",sheet_name='E Comm')


# In[4]:


dt.head()


# In[5]:


dt.info()


# In[6]:


data=dt.copy()


# ### EDA & Feature Engineering
# 

# Checking columns, their datatype & renaming any of them, if necessary

# In[9]:


data.columns


# In[10]:


data.dtypes[data.dtypes!='object'].index


# In[11]:


data.dtypes[data.dtypes=='object'].index


# In[12]:


data.rename(columns={'PreferedOrderCat':'PreferredOrderCategory',
                     'OrderAmountHikeFromlastYear':'OrderCountHikeFromLastYear'},inplace=True)


# Checking unique values of each column¶
# 

# In[14]:


def unique(col):
    """This function returns distinct values of the given column (arguement - 'col') of dataframe 'data'"""
    x=data[col].unique()
    return x


# In[15]:


# to confirm if the data has not any duplicate customer IDs
data['CustomerID'].duplicated().sum()


# In[16]:


unique('PreferredLoginDevice')


# In[17]:


# In column 'PreferredLoginDevice', Phone & Mobile Phone are same. Therefore, they should be merged into one login device.


# In[18]:


unique('CityTier')


# In[19]:


unique('PreferredPaymentMode')


# In[20]:


# In column 'PreferredPaymentMode', CC & COD are short forms of Credit Card & Cash on Delivery respectively. Therefore,
# preferred payment modes CC & COD should be merged into Credit Card & Cash on Delivery respectively


# In[21]:


unique('Gender')


# In[22]:


unique('PreferredOrderCategory')


# In[23]:


# In column 'PreferredOrderCategory', Mobile & Mobile Phone are same. Therefore, they should be merged into one category.


# In[24]:


unique('SatisfactionScore')


# In[25]:


unique('MaritalStatus')


# In[26]:


unique('Complain')


# ### Fixing values of those columns having same values with different names¶
# 

# In[28]:


data['PreferredLoginDevice']=data['PreferredLoginDevice'].replace(['Mobile Phone','Phone'], 'Mobile')


# In[29]:


data['PreferredPaymentMode']=data['PreferredPaymentMode'].replace({'CC':'Credit Card','COD':'Cash on Delivery'})


# In[30]:


data['PreferredOrderCategory']=data['PreferredOrderCategory'].replace('Mobile','Mobile Phone')


# ### Univariate Analysis¶
# 

# In[32]:


def valuecount(col):
    """This function returns frequency distribution (in %) of all distinct values of the given column (arguement - 'col') 
    of dataframe 'data' in tabular form with its visual representation"""
    y=data[col].value_counts(dropna=False,normalize=True)
    z=pd.DataFrame(y).reset_index()
    z.columns=[col,'Proportion']
    z.set_index([col],inplace=True)
    data[col].value_counts(dropna=False).plot(kind='pie')
    plt.plot()
    return z


# In[33]:


valuecount('PreferredLoginDevice')


# In[34]:


valuecount('CityTier')


# In[35]:


valuecount('PreferredPaymentMode')


# In[36]:


valuecount('Gender')


# In[37]:


valuecount('PreferredOrderCategory')


# In[38]:


valuecount('SatisfactionScore')


# In[39]:


valuecount('MaritalStatus')


# In[40]:


valuecount('Complain')


# ### Bivariate Analysis
# 

# CrossTab

# In[43]:


def crosstab(x,y,z):
    """This function performs analysis between two given categorical columns (arguements - 'x' & 'y') of dataframe 'data' 
    in percentage form (arguement - 'z') accordingly"""
    ct=pd.crosstab(data[x],data[y],normalize=z)
    return ct


# In[44]:


crosstab('CityTier','PreferredLoginDevice','index')


# In[45]:


crosstab('CityTier','PreferredPaymentMode','index')


# In[46]:


crosstab('CityTier','PreferredOrderCategory','index')


# In[47]:


crosstab('Gender','MaritalStatus','index')


# In[48]:


crosstab('Gender','CityTier','columns')


# In[49]:


crosstab('Gender','PreferredOrderCategory','index')


# In[50]:


crosstab('PreferredLoginDevice','PreferredPaymentMode',True)


# In[51]:


crosstab('Complain','CityTier','columns')


# In[52]:


crosstab('CityTier','Churn','index')


# In[53]:


crosstab('Gender','Churn','index')


# In[54]:


crosstab('PreferredOrderCategory','Churn','index').round(2)


# In[55]:


crosstab('SatisfactionScore','Churn','index')


# In[56]:


crosstab('Complain','Churn','index')


# Aggregation through Grouping
# 

# In[58]:


pd.DataFrame(data.groupby(['CityTier'])['SatisfactionScore'].mean())


# In[59]:


pd.DataFrame(data.groupby(['CityTier'])['WarehouseToHome'].mean())


# In[60]:


pd.DataFrame(data.groupby(['CityTier'])['HourSpendOnApp'].mean())


# In[61]:


pd.DataFrame(data.groupby(['CityTier'])['OrderCountHikeFromLastYear'].mean())


# In[62]:


pd.DataFrame(data.groupby(['CityTier'])['CouponUsed'].sum())


# In[63]:


pd.DataFrame(data.groupby(['CityTier'])['OrderCount'].sum())


# ### Converting Data into Machine Learning Readable Format¶
# 

# In[65]:


data_ml=data.copy()


# ## Missing Value Treatment¶
# 

# In[67]:


data_ml.isnull().sum()


# In[68]:


# to check the percentage of missing values in the dataframe
(data_ml.shape[0]-data_ml.dropna().shape[0])/data_ml.shape[0]


# In[69]:


data_ml.dtypes


# In[70]:


def mean_fill(col):
    """This function fills missing values of the given numerical column (arguement - 'col') of dataframe 'data_ml' by its mean"""
    data_ml[col].fillna(round(data_ml[col].mean(),2),inplace=True)


# In[71]:


mean_fill('Tenure')
mean_fill('WarehouseToHome')
mean_fill('HourSpendOnApp')
mean_fill('OrderCountHikeFromLastYear')


# In[72]:


def median_fill(col):
    """This function fills missing values of the given numerical column (arguement - 'col') of dataframe 'data_ml' by its median"""
    data_ml[col].fillna(data_ml[col].median(),inplace=True)


# In[73]:


median_fill('CouponUsed')
median_fill('OrderCount')
median_fill('DaySinceLastOrder')


# In[74]:


data_ml.isnull().sum()


# ## Exporting Data for Further Analysis in SQL (after Missing Value Treatment)
# 

# In[76]:


data_churn=data_ml.copy()


# In[77]:


data_churn.to_csv(r"C:\Users\Pooja Nimbalkar\Desktop\Data Analysis\Python Projects\Project1\May-2022.csv", index=False)


# ## Multi-Collinearity Treatment
# 

# Multivariate Analysis
# 

# In[80]:


plt.figure(figsize=(12,6))
cr=data_ml[data_ml.dtypes[data_ml.dtypes!='object'].index].corr()
sns.heatmap(cr,annot=True)
plt.show()


# ## Outlier Treatment
# 

# In[82]:


data_ml.describe(percentiles=[.003,.01,.02,.03,.04,.05,.95,.96,.97,.98,.99,.997]).T


# In[83]:


obj_data=data_ml[data_ml.dtypes[data_ml.dtypes=='object'].index]
num_data=data_ml[data_ml.dtypes[data_ml.dtypes!='object'].index]


# In[84]:


def outlier_min_cap(x):
    """This function performs lower side outlier capping at 0.3% for the given column (arguement - 'x') of dataframe 'num_data'"""
    num_data[x]=np.where(num_data[x]<num_data[x].quantile(.003),num_data[x].quantile(.003),num_data[x])


# In[85]:


outlier_min_cap('CashbackAmount')


# In[86]:


def outlier_max_cap(x):
    """This function performs upperr side outlier capping at 99.7% for the given column (argument - 'x') of dataframe 'num_data'"""
    num_data[x]=np.where(num_data[x]>num_data[x].quantile(.997),num_data[x].quantile(.997),num_data[x])
    return x


# In[87]:


outlier_max_cap('Tenure')
outlier_max_cap('WarehouseToHome')
outlier_max_cap('NumberOfAddress')
outlier_max_cap('DaySinceLastOrder')


# In[88]:


finaldata=pd.concat([obj_data,num_data],axis=1)


# In[89]:


finaldata.head()


# In[90]:


finaldata.drop(columns = 'CustomerID', inplace=True)


# ### Checking and Treating Data Imbalance using SMOTE
# 

# In[92]:


finaldata['Churn'].value_counts(dropna=False)


# In[93]:


add=finaldata[(finaldata['Churn']==1)]


# In[94]:


finaldata_ml=pd.concat([finaldata,add,add,add,add])


# In[95]:


finaldata_ml['Churn'].value_counts(dropna=False)


# #### Creating Dummies of all Categorical Columns having non-binary format
# 

# In[97]:


finaldata_ml.head()


# In[98]:


mldata=pd.get_dummies(data=finaldata_ml,columns=['PreferredLoginDevice','PreferredPaymentMode',
                                                 'Gender','PreferredOrderCategory','MaritalStatus','CityTier'],
                        drop_first=True)


# In[ ]:




