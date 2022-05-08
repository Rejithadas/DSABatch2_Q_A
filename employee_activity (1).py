#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[68]:


data=pd.read_csv("C:/Users/Lenovo/OneDrive/Desktop/DSA_ActivityDataset/employee.csv")
data


# In[69]:


#Load the data into the pandas environment and identify some basic details of the dataset.
data.info()


# In[36]:


data.columns


# In[92]:


#Reset the index as "name"
data=pd.read_csv("C:/Users/Lenovo/OneDrive/Desktop/DSA_ActivityDataset/employee.csv",index_col="name")
data


# In[93]:


#Select rows for specific names Jack Morgan and Josh wills.
data.loc[["Jack Morgan","Josh Wills"]]


# In[94]:


#Select data for multiple values "Sales" and “Finance”
data[data["department"].isin(["Sales", "Finance"])]


# In[95]:


#Display employee who has more than 700 performance score.
data[data['performance_score']>700]


# In[96]:


#Display employee who has more than 500 and less than 700 performance score
data[(data["performance_score"]>500)&(data["performance_score"]<700)]


# In[97]:


#Check and handle missing values in the dataset
data.isna().sum()


# In[98]:


data.dtypes


# In[99]:


fg=data[["age","income"]]
fg.hist(figsize=(15,10))
plt.show()


# In[100]:


data["age"]=data["age"].fillna(data["age"].median())
data.isna().sum()


# In[101]:


data["income"]=data["income"].fillna(data["income"].median())
data.isna().sum()


# In[102]:


data["gender"]=data["gender"].fillna(data["gender"].mode()[0])
data.isna().sum()


# In[103]:


#Check the outliers and handle outliers in performance score using Percentiles
plt.boxplot(data["performance_score"])   
plt.show()


# In[105]:


Q1=np.percentile(data["performance_score"],25,interpolation="midpoint")
Q2=np.percentile(data["performance_score"],50,interpolation="midpoint")
Q3=np.percentile(data["performance_score"],75,interpolation="midpoint")
print(Q1,Q2,Q3)


# In[106]:


IQR=Q3-Q1
lw_lim=Q1-1.5*IQR
up_lim=Q3+1.5*IQR
print(lw_lim,up_lim)


# In[107]:


outliers=[]
for x in data["performance_score"]:
    if (x>up_lim)or(x<lw_lim):
        outliers.append(x)
outliers


# In[108]:


data=pd.get_dummies(data,columns=["gender"])
data


# In[109]:


#Do the standard scaling on the feature performance score
data["performance_score"]


# In[110]:


from sklearn  import preprocessing
S =preprocessing. StandardScaler()
Std_S = pd.DataFrame(data["performance_score"])
Std_S = S.fit_transform(Std_S)
Std_S = pd.DataFrame(Std_S)
Std_S.describe()


# In[ ]:




