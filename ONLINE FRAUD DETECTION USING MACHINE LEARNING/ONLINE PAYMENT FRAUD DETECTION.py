#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("fraud.csv")


# In[3]:


df.head(15)


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df['isFlaggedFraud'].value_counts()


# In[8]:


df['isFraud'].value_counts()


# In[9]:


df['step'].value_counts()


# In[10]:


df['type'].value_counts()


# In[11]:


df['oldbalanceDest'].value_counts()


# In[12]:


df['newbalanceDest'].value_counts()


# In[13]:


df.drop('isFlaggedFraud', inplace=True, axis=1)


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[15]:


df.drop('step', inplace=True, axis=1)


# In[16]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


# In[17]:


df['type'] = label_encoder.fit_transform(df['type'])
df['nameOrig'] = label_encoder.fit_transform(df['nameOrig'])
df['nameDest'] = label_encoder.fit_transform(df['nameDest'])


# In[18]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f')


# In[19]:


df.drop(['nameOrig','nameDest'], inplace=True, axis=1)


# In[20]:


from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()


# In[21]:


X = df.drop('isFraud',axis=1)
y = df['isFraud']


# In[22]:


X = scalar.fit_transform(X)


# In[23]:


from sklearn.model_selection import train_test_split as tt
X_train,X_test,y_train,y_test = tt(X,y,test_size=0.2,random_state=0)


# In[24]:


from sklearn.linear_model import LogisticRegression as LR
lr = LR()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# In[25]:


y_pred


# In[26]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[27]:


from sklearn.tree import DecisionTreeClassifier as DTC
model = DTC()


# In[28]:


model.fit(X_train, y_train)
y_model = model.predict(X_test)


# In[29]:


y_model


# In[30]:


accuracy_score(y_test, y_model)


# In[ ]:




