
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd



# In[2]:


os.getcwd()
os.chdir("E:\Data Science\Capstone Projects\Capstone Project_Health_care_ML")


# In[3]:


dataset=pd.read_csv("train Data.csv")
dataset.drop('id',axis=1,inplace=True)
dataset.head(1)


# In[4]:


output=pd.read_csv("train labels.csv")
output.drop('id',axis=1,inplace=True)
#type(y_train.service_a)
#type(cols.id)
output.head()


# In[62]:


#dataset1=dataset.dropna(how='all')--No rows has full nas
#dataset1


# In[5]:


col=list(dataset.columns)
numcol = []
catcol = []
ordcol = []
for i in col:
    if list(i)[0] =='n':
        numcol.append(i)
    elif list(i)[0] =='c':
        catcol.append(i)
    elif list(i)[0] =='o' :
        ordcol.append(i)


# In[6]:


numcol1=dataset[numcol]
catcol1=dataset[catcol]
ordcol1=dataset[ordcol]
#numcol1.head()--116
#catcol1.head(1)--1050
ordcol1.head(1)--211
#list(col)[0][0]
#text="king"
#list(text)[0]
#type(text)


# In[8]:


#Filling nas in numerical columns
#numcol1.head(2)
numcol1.fillna(numcol1.median(),inplace=True)
numcol1.isnull().sum().sum()
numcol1.drop('n_0101',axis=1,inplace=True)


# In[15]:


#Filling nas in Categorical columns
#catcol1[catcol]=catcol1[catcol].fillna(catcol1.mode().iloc[0])
#catcol1.isnull().sum().sum()
catcol1 = catcol1.apply(lambda x:x.fillna(x.value_counts().index[0]))
# Drop the columns which have all the values as nan shown in the below cells.
#catcol1.dtypes
#catcol1.head()
#catcol1.c_0491


# In[12]:


#Checking if there are any columns with entire missing values..

col1=list(catcol1.columns)
colna=[]
for i in col1:
    if catcol1[i].isnull().sum() ==14644:
        colna.append(i)

colna
#catcol1.isnull().sum().sum()


# In[13]:


catcol1.drop(catcol1[colna],axis=1,inplace=True)


# In[18]:


#Filling nas in Ordinal columns
ordcol1[ordcol]=ordcol1[ordcol].fillna(ordcol1.mode().iloc[0])
#ordcol1.isnull().sum().sum()


# In[20]:


data1=dataset.iloc[:,0:1]
data1.head()
#data1.head()()


# In[22]:


#Join data1, categorical columns, numerical and ordinal
#data1,numcol1,catcol1,ordcol1
cols=data1.join(numcol1,how='outer')
cols=cols.join(catcol1,how='outer')
cols=cols.join(ordcol1,how='outer')
cols.head()
#cols.dtypes
cols['release']=cols['release'].astype('category')


# In[25]:


# Convert all the categorical columns to category data type
#cols.dtypes
cols[cols.select_dtypes(['object']).columns] = cols.select_dtypes(['object']).apply(lambda x: x.astype('category'))


# In[27]:


#Applying dummies on categorical variables
cols2=pd.get_dummies(cols,drop_first=True)

#cols2.shape


# In[23]:


from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier


#classifier = ClassifierChain(GaussianNB())



# In[34]:


X_train, X_test, y_train, y_test = train_test_split(cols2,output, 
                                                    test_size=0.3,
                                                    random_state=0)


# In[35]:


classifier = LabelPowerset(GaussianNB())



# In[36]:


# train
classifier.fit(X_train,y_train)
#y_train.head()
#y_train.isnull().sum()
#cols.isnull().sum()


# In[38]:


# predict
predictions1 = classifier.predict(X_test)


#predictions = classifier.predict(X_test)



# In[49]:


from sklearn.metrics import accuracy_score,classification_report
accuracy_score(y_test,predictions1)


# In[50]:


print(classification_report(y_test,predictions1))


# # Accuracy is too less so, drop few columns at first place where there are more nas i.e., 20% nas

# In[7]:


dataset2=dataset.drop('release',axis=1)
c=dataset2.columns
t=14644


# In[8]:


d=[]
for i in list(c):
    if t - dataset2[i].isnull().sum() <2800 :
        d.append(i)
    


# In[12]:


dataset2.drop(dataset2[d],axis=1,inplace=True)
dataset2.shape


# In[13]:


col=list(dataset2.columns)
numcol = []
catcol = []
ordcol = []
for i in col:
    if list(i)[0] =='n':
        numcol.append(i)
    elif list(i)[0] =='c':
        catcol.append(i)
    elif list(i)[0] =='o' :
        ordcol.append(i)


# In[14]:


numcol1=dataset[numcol]
catcol1=dataset[catcol]
ordcol1=dataset[ordcol]
numcol1.head()#34
#catcol1.head(1)#274
#ordcol1.head(1)#40


# In[15]:


#Filling nas in numerical columns
#numcol1.head(2)
numcol1.fillna(numcol1.median(),inplace=True)
numcol1.isnull().sum().sum()



# In[16]:


catcol1 = catcol1.apply(lambda x:x.fillna(x.value_counts().index[0]))
catcol1.isnull().sum().sum()


# In[17]:


#Filling nas in Ordinal columns
ordcol1[ordcol]=ordcol1[ordcol].fillna(ordcol1.mode().iloc[0])
ordcol1.isnull().sum().sum()


# In[18]:


data1=dataset.iloc[:,0:1]
data1.head()
#data1.head()()


# In[19]:


#Join data1, categorical columns, numerical and ordinal
#data1,numcol1,catcol1,ordcol1
cols=data1.join(numcol1,how='outer')
cols=cols.join(catcol1,how='outer')
cols=cols.join(ordcol1,how='outer')
cols.head()
#cols.dtypes
cols['release']=cols['release'].astype('category')


# In[20]:


# Convert all the categorical columns to category data type
#cols.dtypes
cols[cols.select_dtypes(['object']).columns] = cols.select_dtypes(['object']).apply(lambda x: x.astype('category'))


# In[21]:


#Applying dummies on categorical variables
cols2=pd.get_dummies(cols,drop_first=True)

cols2.shape


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(cols2,output, 
                                                    test_size=0.3,
                                                    random_state=1)


# In[90]:


classifier = ClassifierChain(GaussianNB())


# In[91]:


classifier.fit(X_train,y_train)


# In[92]:


predictions1 = classifier.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score,classification_report
accuracy_score(y_test,predictions1)


# In[27]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# In[29]:


classif = OneVsRestClassifier(SVC(kernel='linear'))    


# In[ ]:


classif.fit(X_train,y_train)

