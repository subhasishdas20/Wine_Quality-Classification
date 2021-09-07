#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv(r"C:\Learning\python_class\winequality-red.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info


# In[6]:


df.describe()


# ##visualization

# In[7]:


#correlation map to see correlation between variable 
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot =True)


# In[8]:


#quality vs sulphate 
sns.barplot(x="quality",y="alcohol",data = df)


# In[9]:


#quality vs sulphate 
sns.barplot(x="quality",y="volatile acidity",data = df)


# In[10]:


sns.pairplot(df)


# In[11]:


df["quality"].value_counts()


# ## Converting Quality Scale into Two Possible Outcomes

# In[12]:


qua = []
for i in df["quality"]:
  if i >6:
    qua.append(1)

  else:
    qua.append(0)


# In[13]:


df["quality"] = qua


# In[14]:


df.head()


# In[15]:


df["quality"].value_counts()


# In[16]:


sns.barplot(x = "quality",y="alcohol",data = df)


# ## Segregatting  X & Y 

# In[17]:


x = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[18]:


x.head


# In[19]:


y.head


# ## Splitting Dataset

# In[20]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2)


# In[21]:


len(x_train) , len(x_test)


# ## Scaling 

# In[22]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[26]:


from sklearn.svm import  SVC
cls = SVC(kernel = "linear")
cls.fit(x_train,y_train)


# In[27]:


y_pred = cls.predict(x_test)
y_pred


# In[28]:


y_test


# ## Metrics

# In[29]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize = (7,5))
sns.heatmap(cm,annot = True, fmt= "2.0f")


# In[30]:


accuracy_score(y_pred,y_test)


# In[ ]:





# In[ ]:




