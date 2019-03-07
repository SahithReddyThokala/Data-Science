
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


os.chdir("E:\Data Science\Handson")


# In[11]:


#os.getcwd()
#os.chdir("E:\Data Science\Use cases")


# In[48]:


df=pd.read_csv("Churn_MV.csv")


# In[13]:


df.head()
df.describe()


# In[14]:


df.isnull().sum()


# In[15]:


df1=df.drop(['State','Area Code','Phone'],axis=1)


# In[16]:


df1=df1.dropna(how='all')
df1.reset_index(drop=True, inplace=True)
df1=df1.drop('VMail Plan',axis=1)#Highly corelated


# In[17]:


df1.isnull().sum()
#df1.head()


# In[18]:


#Checking class balance
#df['Churn'].value_counts()
df.Churn.value_counts()
#df.Churn.value_counts('Intl Calls')


# Handling Missing Info

# In[19]:


#import warnings
df1.fillna(df1.mean(),inplace=True)
#df1.dtypes


# In[11]:


#Replace with Mean
df1['VMail Message']=df1['VMail Message'].astype('category')
#df1['Churn']=df1['Churn'].astype('float')
df1['Intl Plan']=df1['Intl Plan'].astype('category')

df1.isnull().sum()
#df1.describe()
#df1['Day Charge']-df1['Daily Charges MV']


# Seperate input and output

# In[12]:


inputs=df1.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17]]
output=df1.iloc[:,7]


# In[13]:


#inputs


# In[13]:


catcols=inputs.iloc[:,[1,7]]
numcols=inputs.iloc[:,[0,2,3,4,5,6,8,9,10,11,12,13,14,15,16]]


# In[14]:


output.head()


# In[15]:


catcols.head()
numcols.head(5)


# # Corelation Plots

# In[17]:


df.corr()#VMailPlan,

#numcols.dtypes
#--Day Mins, Day Charge,Daily Charges MV
#--Eve Mins, Eve Charge
#--Night Mins, Night Charge
#--Intl Mins, Intl Charge


# In[49]:


numcols.dtypes
numcolsnames=['Account Length','Day Mins','Eve Mins','Night Mins','Intl Mins','CustServ Calls','Day Calls',             'Day Charge','Daily Charges MV','Eve Calls','Eve Charge','Night Calls','Night Charge',             'Intl Calls','Intl Charge']
numcols.columns


# In[19]:


#numcols.loc[:,(numcols.dtypes=='float64') ].hist(figsize=[11,11])
numcols.hist(figsize=[11,11])


# In[20]:


#plt.scatter(df['VMail Message'], df['Churn'])
#plt.hist(df['VMail Message'],df['VMail Plan'])
#plt.hist(df['VMail Message'])
#plt.xlabel('VMail Message')
#plt.show()
#plt.scatter(df['VMail Plan'], df['Churn'])


# In[21]:


numcols.boxplot(column=numcolsnames,figsize=[20,20])


# In[22]:


#df.corr()
#plt.matshow(df.corr())
numcols.boxplot(numcolsnames, sym='gx', notch=False,figsize=[20,20])



# In[23]:


#plt.boxplot(df1['Account Length'], sym='gx', notch=False)
#help(plt.boxplot)


# # Handling Outliers

# In[24]:


x=numcols['Account Length'].quantile([0.25,0.5,0.75])
print(x.values)


# In[25]:


IQR=x[0.75]-x[0.25]
IQR15=IQR * 1.5
IQR15


# In[26]:


U_W=x[.75]+IQR15
L_W=x[.25]-IQR15
print(U_W,L_W)


# In[27]:


print("Upper Outliers")
numcols['Account Length'][numcols['Account Length'] >U_W]


# In[28]:


print("Lower Outliers")
numcols['Account Length'][numcols['Account Length'] < L_W]


# In[29]:


W = numcols['Account Length'].quantile([0.05,0.95])
W


# In[30]:


numcols['Account Length'][numcols['Account Length'] >U_W]= W[0.95]


# In[31]:


print('Upper outliers')
numcols['Account Length'][numcols['Account Length'] >U_W]
#plt.boxplot(numcols['Account '])


# In[16]:


numcolsnames=['Account Length','Day Mins','Eve Mins','Night Mins','Intl Mins','CustServ Calls','Day Calls',             'Day Charge','Daily Charges MV','Eve Calls','Eve Charge','Night Calls','Night Charge',             'Intl Calls','Intl Charge']
for i in numcolsnames:
    x=numcols[i].quantile([0.25,0.5,0.75])
    IQR=x[0.75]-x[0.25]
    IQR15=IQR * 1.5
    U_W=x[.75]+IQR15
    L_W=x[.25]-IQR15
    W = numcols[i].quantile([0.05,0.95])
    numcols[i][numcols[i] >U_W]= W[0.95]
    numcols[i][numcols[i] <L_W]= W[0.05]


# In[33]:


numcols.boxplot(numcolsnames, sym='gx', notch=False,figsize=[20,20])
#help(plt.boxplot)
#plt.boxplot(numcols['Account Length'], sym='gx', notch=False)
#numcols
#numcols.describe()


# In[34]:


numcolsnames=['Account Length']#,'Day Mins','Eve Mins','Night Mins','Intl Mins','CustServ Calls','Day Calls',\
             #'Day Charge','Daily Charges MV','Eve Calls','Eve Charge','Night Calls','Night Charge',\
             #'Intl Calls','Intl Charge']
for i in numcolsnames:
    x=numcols[i].quantile([0.25,0.5,0.75])
    print(x)
    IQR=x[0.75]-x[0.25]
    IQR15=IQR * 1.5
    print(IQR15)
    U_W=x[.75]+IQR15
    L_W=x[.25]-IQR15
    print(U_W,L_W)
    W = numcols[i].quantile([0.05,0.95])
    print(W)
    numcols[i][numcols[i] >U_W]= W[0.95]
    numcols[i][numcols[i] <L_W]= W[0.05]


# In[35]:


#numcols['Account Length']
plt.boxplot(numcols['Account Length'], sym='gx', notch=False)


# # EDA

# In[36]:


# Impact of Intl calls and Intl Plan on Churn
import seaborn as sns
fig, ax=plt.subplots(figsize=(8,6))

sns.countplot(x='Churn', data=df1, hue='Intl Plan')
ax.set_ylim(0,500)
plt.title("Impact of Intl Plan on Churn")


# In[37]:


fig, ax=plt.subplots(figsize=(8,8))
sns.countplot(x='Churn',data=df,hue='VMail Plan')
#ax.set_ylim(0,)
plt.title("Impact of VMail Plan on Churn")


# In[38]:


plt.scatter(df['Day Calls'], df['Churn'])
plt.xlabel('Day Calls')
plt.ylabel('Churn')
plt.show()
plt.scatter(df['Day Mins'], df['Churn'])



# In[39]:


df.hist(figsize=[10,10])


# In[40]:


df.plot(kind='density',subplots=True,sharex=False,layout=(7,3))


# In[41]:


#numcols.loc[:,(numcols.dtypes=='float64') ].hist(figsize=[11,11])
df.hist('Account Length','Churn')
df.hist('VMail Message')
#plt.hist[df1['Account Length']]


# In[42]:


#df.plot(x='Churn',y=['VMail Message','Intl Plan','Account Length'],kind='Bar',width=1)


# In[43]:


df.plot(x='Intl Plan',y='Churn',kind='Bar',width=1,stacked=True)


# In[44]:


#plt.bar(index,df['Intl Plan'],bar_width,color='b',label='Intl Plan')
#plt.bar(index,df['Churn'],bar_width,color='c',label='Churn')
#rects1 = plt.bar(index, means_frank, bar_width,
#                 alpha=opacity,
 #                color='b',
 #                label='Frank')
#df[['Intl Plan','Churn']].plot(kind='bar', title ="Comp", figsize=(15, 10), legend=True, fontsize=12,width=1)


# In[45]:


plt.scatter(df['Eve Mins'], df['Churn'])
plt.scatter(df['Eve Charge'],df['Churn'])
plt.legend(('Eve Mins','Eve Charge'))
#Drop Eve Charge
#Day Mins, Day Charge,Daily Charges MV
#--Eve Mins, Eve Charge
#--Night Mins, Night Charge
#--Intl Mins, Intl Charge


# In[46]:


plt.scatter(df['Day Mins'],df['Churn'])
plt.xlabel('Day Mins')
plt.ylabel('Churn')
plt.scatter(df['Day Charge'],df['Churn'])
plt.legend(('Day Mins','Day Charge'))
#Drop Day Charge


# In[47]:


plt.scatter(df['Night Mins'],df['Churn'])
plt.xlabel('Night Mins')
plt.ylabel('Churn')
plt.scatter(df['Night Charge'],df['Churn'])
plt.legend(('Night Mins','Night Charge'),fontsize=8)
#Drop Night Charge


# In[48]:


plt.scatter(df['Intl Mins'],df['Churn'])

plt.xlabel('Intl Mins')
plt.ylabel('Churn')
plt.scatter(df['Intl Charge'],df['Churn'])
plt.legend(('Intl Mins','Intl Charge'),fontsize=10)
#Drop Intl Charge


# # Dropping Highly corelated columns after comparing its effect on target variable

# In[17]:


numcols2=numcols.drop(['Intl Charge','Night Charge','Day Charge','Eve Charge'],axis=1)
numcols2=pd.DataFrame(numcols2)
numcols2.head()
#numcols.columns


# In[50]:


#
#numcols1.columns=numcols1names
#numcols1.head(2)
numcols1.head(1)


# # Adding categorical columns to the dataset and applying dummies on catcols

# In[18]:


from sklearn import preprocessing
cols=preprocessing.normalize(numcols2)
cols=pd.DataFrame(numcols2)
#numcols1.columns=numcols1names
#normalized_X = preprocessing.normalize(X)
cols.head()
#numcols1.describe()
#numcols1=minmaxscaling
#cols.describe()


# In[19]:


cols=cols.join(catcols,how='outer')
cols.dtypes
cols.head()


# In[20]:


colsnames=['Account Length','Day Mins','Eve Mins','Night Mins','Intl Mins','CustServ Calls','Day Calls',               'Daily Charges MV','Eve Calls','Night Calls',                 'Intl Calls','VMail Message','Intl Plan']
cols.columns=colsnames
cols.head()



# In[21]:


cols.dtypes
cols['VMail Message']=cols['VMail Message'].astype('category')
cols['Intl Plan']=cols['Intl Plan'].astype('category')


# In[22]:


cols=pd.get_dummies(cols,prefix=['VMail Message','Intl PLan'],drop_first=True)
cols.head(2)
#cols=numcols.join(catcols,how='outer')
#cols.head()


# In[ ]:



#Churn_data=cols.join(output,how='outer')
#Churn_data.head()


# In[ ]:


#Churn_data['Churn'].value_counts()
#output


# # Building the model before before handling Class Imbalance i.e., Oversampling

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(cols,output, 
                                                    test_size=0.3,
                                                    random_state=0)
#X_train
#y_train


# In[25]:


model=LogisticRegression()
model=model.fit(X_train,y_train)


# In[26]:


preds = model.predict(X_test)
#y_pred-- Arrays
preds[1]


# In[27]:


predsTrain=model.predict(X_train)


# In[ ]:


probs=model.predict_proba(X_test)
probs[1]


# In[ ]:


#print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# In[28]:


print(classification_report(y_test,preds))


# In[29]:


print(classification_report(y_train,predsTrain))


# # To find the (columns) co efficients impact on the model

# In[ ]:


import numpy as np
import statsmodels.api as sm
from scipy import stats
import pandas as pd


# In[ ]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(numcols1,output, 
                                                    test_size=0.3,
                                                    random_state=1)
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
#X_train1.dtypes
#X_train1['VMail Message'].astype(float)
#cols['VMail Message']=cols['VMail Message'].astype(float)
#cols['Intl Plan']=cols['Intl Plan'].astype(float)
#cols.dtypes
#output=output.astype(float)
#output.head()
#y_train1.dtypes
#y_test1.dtypes
#y_test1


# In[ ]:






# In[ ]:


#model1=sm.Logit(y_train.astype(float),X_train.astype(float))
y_train1=list(y_train1)
model1=sm.Logit(y_train1,X_train1)
model1=model1.fit()
#y_train.dtype
#X_test1.head()
#X_test1.dtypes
#y_test1.head()
#X_pred1= model1.predict(X_train1)
#X_pred1
#
#print("Accuracy: ", metrics.accuracy_score(X_test1, X_pred1))


# In[ ]:


print(model1.summary())


# In[ ]:


Churn_count = df1.Churn.value_counts()
df1_class_1=df1[df1.Churn==1]
df1_class_0=df1[df1.Churn==0]
print(df1_class_1)
count_class_0=Churn_count[0]
print('Class 0:', Churn_count[0])
print('Class 1:', Churn_count[1])
print('Proportion:', round(Churn_count[0] / Churn_count[1], 2), ': 1')

Churn_count.plot(kind='bar', title='Count (Churn)');


# # Applying over sampling to handle class imbalance

# In[30]:


sampledcols=cols.join(output,how='outer')
sampledcols.head(2)
#sampledcols=cols


# In[31]:


Churn_count = sampledcols.Churn.value_counts()
class_1=sampledcols[sampledcols.Churn==1]
class_0=sampledcols[sampledcols.Churn==0]

count_class_0=Churn_count[0]
print(count_class_0)
print('Class 0:', Churn_count[0])
print('Class 1:', Churn_count[1])
print('Proportion:', round(Churn_count[0] / Churn_count[1], 2), ': 1')

Churn_count.plot(kind='bar', title='Count (Churn)');


# In[32]:


sampledcols_class1_over=class_1.sample(count_class_0,replace=True)
#print(sampledcols_class1_over)
sampledcols_over=pd.concat([class_0,sampledcols_class1_over],axis=0)
#df1_class_1_over = df1_class_1.sample(count_class_0, replace=True)
#df1_test_over = pd.concat([df1_class_0, df1_class_1_over], axis=0)

print('Random over-sampling:')
print(sampledcols_over.Churn.value_counts())

sampledcols_over.Churn.value_counts().plot(kind='bar', title='Count (Churn)');
#cols
sampledcols_over.head()


# # Seperating target variable i.e., Churn

# In[33]:


#Churn=sampledcols_over.head(2)
y=sampledcols_over['Churn']
X=sampledcols_over.drop(['Churn'],axis=1)
#X.head(1)
#X.head()
#y.head(1)


# In[34]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)
#Xo_train.head()
#yo_train.head()


# In[35]:


model=LogisticRegression()
model=model.fit(X_train,y_train)


# In[36]:


preds=model.predict(X_test)


# In[37]:


predst=model.predict(X_train)


# # Getting clarity between actuals and predicted values

# In[38]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(preds,y_test))


# In[ ]:


print(confusion_matrix(y_test,preds))


# # Check if the model is over fitting

# In[39]:


print(classification_report(y_test,preds))


# In[40]:


print(classification_report(y_train,predst))


# In[ ]:


#print(y_test,preds)


# # Check the model using crossval

# In[ ]:


from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import cross_val_score,cross_validate


# In[ ]:


#scoring = {'acc': 'accuracy',
  #         'prec': 'precision',
 #          'rec': 'recall'}
#scores=cross_validate(model,X,y,cv=10,n_jobs=-1,scoring=scoring)
#print(scores.keys())
#print(scores['test_acc']) 
print ("Recall :")
cross_val_score(model,X,y,cv=10,n_jobs=-1,scoring='recall',).mean()






# In[ ]:


print ("Precision :")
cross_val_score(model,X,y,cv=10,n_jobs=-1,scoring='precision',).mean()


# In[ ]:


print ("f1 :")
cross_val_score(model,X,y,cv=10,n_jobs=-1,scoring='f1',).mean()


# In[ ]:


print ("Accuracy :")
cross_val_score(model,X,y,cv=10,n_jobs=-1,scoring='accuracy',).mean()


# In[ ]:


#help(cross_val_score)


# # Decision Trees

# In[ ]:


churn=sampledcols_over
churn.head()


# # Seperate Target Variable

# In[ ]:


from sklearn.tree import DecisionTreeClassifier,export_graphviz


# In[ ]:


y=churn['Churn']
X=churn.drop(['Churn'],axis=1)


# In[ ]:


DTC = DecisionTreeClassifier(max_depth=3,min_samples_leaf=4,
                             class_weight='balanced',
                            min_samples_split=10)
#Tune the hyper parameters for better metrics.Lets do that in Random forest model.


# In[ ]:


#X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.3, random_state=123)--no need of seperating train and test
#sets everytime. We can use the same splits that were created above 


# In[67]:


DTC.fit(X_train,y_train)


# In[ ]:


predsDTC=DTC.predict(X_test)
print(classification_report(y_test,predsDTC))


# In[ ]:


predsDTCT=DTC.predict(X_train)
print(classification_report(y_train,predsDTCT))


# In[ ]:


with open("decisiontree.dot", 'w') as f:
    f = export_graphviz(DTC, out_file=f,feature_names=X.columns.values,filled=True, rounded=True,special_characters=True,class_names=['0','1'], proportion=True)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


RFC=RandomForestClassifier(class_weight='balanced',n_jobs=-1)


# # Grid Search CV

# In[ ]:


param_grid = { 
    'n_estimators': [1000,1500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy'],
    'min_samples_leaf' :[4,5,6,7,8]
}


# In[ ]:


CV_RFC = GridSearchCV(estimator=RFC, param_grid=param_grid, cv= 10)


# In[ ]:


#CV_RFC.fit(X_train,y_train)


# In[ ]:


predsRFC=RFC.predict(X_test)
print(classification_report(y_test,predsRFC))


# In[53]:


predsRFCT=RFC.predict(X_train)
print(classification_report(y_train,predsRFCT))



# # Kmeans clustering

# In[20]:


df.head()


# In[60]:


#clust=df.iloc[:,[20,21]]
clust=df[['Area Code','State']]
clust.head()


# In[61]:


clust=clust.dropna(how='all')
#clust=clust.fillna(clust.mean(),inplace=True)

#
#clust.head()


# In[62]:


#clust.reset_index(drop=True,inplace=True)
type(clust)
clust.dtypes


# In[63]:


clust.head()


# In[68]:


#clust.groupby(['Area Code','State'])
clust1=clust.groupby(["Area Code", "State"],as_index=False).size()
clust1=clust1.to_frame(name = 'size')#.reset_index()
clust1.head()


# In[69]:


clust2=clust1[['size']]


# In[70]:


from sklearn.cluster import KMeans
kmeans=KMeans()
kmeans=kmeans.fit(clust2)


# In[71]:


clusters=kmeans.predict(clust2)


# In[72]:


centroids=kmeans.cluster_centers_
print(centroids)


# In[73]:


#print(clusters)
clusters=pd.DataFrame(clusters)
colname=['cluster']
clusters.columns=colname
clusters.head()


# In[76]:


df_clust=clust1.join(clusters,how='outer')
df_clust.head(2)


# In[49]:


df.head(1)


# # ANN

# In[41]:


churn_ANN=cols


# In[42]:


churn_ANN.head()


# In[73]:


Xc=churn_ANN
#Xc.Churn.head()
yc=output
#yc.head()


# In[74]:


from sklearn.preprocessing import StandardScaler,Normalizer

from sklearn.model_selection import train_test_split


# In[75]:


X_train,X_test,y_train,y_test=train_test_split(Xc,yc,test_size=0.3,random_state=1)


# In[76]:


sc=StandardScaler()
#X_train.sum().isnull()
#np.isnan(X_train).any()


# In[77]:


X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
X_train


# In[75]:


#nz=Normalizer()
#X_train=nz.fit_transform(X_train)
#X_test=nz.fit_transform(X_test)



# In[78]:


#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import Normalizer


# In[79]:


classifier=Sequential()


# In[84]:


classifier.add(Dense(units=16,activation='sigmoid',kernel_initializer='uniform',input_dim=57))


# In[88]:


classifier.add(Dense(units=16,activation='sigmoid',kernel_initializer='uniform'))


# In[89]:


classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))


# In[90]:


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[91]:


classifier.fit(X_train,y_train,epochs=10)


# In[92]:


y_pred = classifier.predict(X_test) > 0.5


# In[50]:


from sklearn.metrics import classification_report


# In[93]:


print(classification_report(y_test,y_pred))


# In[82]:


y_predtrain=classifier.predict(X_train) >0.5


# In[83]:


print(classification_report(y_train,y_predtrain))

