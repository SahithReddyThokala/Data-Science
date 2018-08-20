
# coding: utf-8

# In[1]:


#### Text classifcation model - Naive Bayes


# In[2]:


import os 
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from itertools import islice
from sklearn.naive_bayes import MultinomialNB
import pickle


# In[3]:


os.chdir("/home/visa/Python/News")


# In[4]:


df = pd.read_csv("train.csv", header=None)


# In[5]:


df.columns = ["Class","ShortDesc","Desc"]
df.head()


# ## Add a full list of stopwords

# In[12]:


stops = stopwords.words('english')
ad_stopwords = ['monday', 'like', 'com', 'use', 'help', 'create', 'time', 'want', 'a', 'about', 'above', 'across', 'after', 'afterwards'] 
stops.extend(ad_stopwords)          


# In[8]:


stops[0:10]


# In[8]:


##REGEX FOR DATE REMOVAL
#import re
txt='Monday, April, 4, 2016, 10:16:06'
re1='(Monday)'	# Day Of Week 1
re2='.*?'	# Non-greedy match on filler
re3='(April)'	# Month 1
re4='.*?'	# Non-greedy match on filler
re5='(4)'	# Day 1
re6='.*?'	# Non-greedy match on filler
re7='(2016)'	# Year 1
re8='.*?'	# Non-greedy match on filler
re9='(10:16:06)'	# HourMinuteSec 1

rg = re.compile(re1+re2+re3+re4+re5+re6+re7+re8+re9,re.IGNORECASE|re.DOTALL)
m = rg.search(txt)
if m:
    dayofweek1=m.group(1)
    month1=m.group(2)
    day1=m.group(3)
    year1=m.group(4)
    time1=m.group(5)


# In[9]:


#import re
re1='(b)'	# Variable Name 1
re2='.*?'	# Non-greedy match on filler
re3='(4\\/04\\/2016)'	# DDMMYYYY 1

rd = re.compile(re1+re2+re3,re.IGNORECASE|re.DOTALL)
m = rg.search(txt)
if m:
    var1=m.group(1)
    ddmmyyyy1=m.group(2)


# In[10]:


#import re
txt='\\\\r\\\\r____________________________\\\\r'

re1='(\\\\\\\\r\\\\\\\\r____________________________\\\\\\\\r)'	# Windows UNC 1

rt = re.compile(re1,re.IGNORECASE|re.DOTALL)
m = rg.search(txt)
if m:
    unc1=m.group(1)


# In[6]:


#import nltk
def _remove_noise(input_text):
    input_text = str(input_text).encode('ascii', 'ignore')
    input_text = str(input_text).replace(",", "")
    #input_text = str(input_text).replace("\'\\", "")
    #input_text = str(input_text).replace("\'\", "")
    input_text = re.sub(rg, ' ', input_text)
    #input_text = re.sub([[:punct:]], '', input_text) -- this is a step in R 
    input_text = re.sub(rd, ' ', input_text)
    input_text = re.sub(rt, ' ', input_text)
    words = str(input_text).split()
    pos_words = nltk.pos_tag(words)
    noise_free_words = [i[0] for i in pos_words if i[1] in ('NN')]
    noise_free_words = [word for word in noise_free_words if word.lower() not in stops]
    return noise_free_words


# In[13]:


df["cleaned"] = df.Desc.apply(_remove_noise)
df.head()


# In[14]:


#from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
df['stemmed'] = df.cleaned.map(lambda x: ' '.join([lemmatizer.lemmatize(y) for y in x]))
df.stemmed.head()


# In[15]:


cvec = CountVectorizer(stop_words= stops, min_df=1, max_df=.5, ngram_range=(1,2))


# In[16]:


#from itertools import islice
cvec.fit(df.stemmed)

list(islice(cvec.vocabulary_.items(), 20))


# In[17]:


len(cvec.vocabulary_)


# In[18]:


cvec = CountVectorizer(stop_words=stops, min_df=.001, max_df=.99, ngram_range=(1,2))
cvec.fit(df.stemmed)
len(cvec.vocabulary_)
#pickle.dump(cvec.vocabulary_,open("feature.pkl","wb"))


# In[19]:


cvec_counts = cvec.transform(df.stemmed)
print ('sparse matrix shape:', cvec_counts.shape)
print ('nonzero count:', cvec_counts.nnz)
print ('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))


# In[20]:


transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(cvec_counts)
transformed_weights

#transformer = TfidfTransformer()
#loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
#tfidf = transformer.fit_transform(loaded_vec.fit_transform(cvec_counts))


# In[21]:


weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(20)


# In[23]:


target = df["Class"]


# In[22]:


nb = MultinomialNB()


# In[24]:


nb.fit(transformed_weights, target)


# In[25]:


df1 = pd.read_csv("test.csv", header=None)


# In[26]:


df1.columns = ["Class","ShortDesc","Desc"]
df1.head()


# In[27]:


df1["cleaned"] = df1.Desc.apply(_remove_noise)
df1.head()


# In[28]:


#from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
df1['stemmed'] = df1.cleaned.map(lambda x: ' '.join([lemmatizer.lemmatize(y) for y in x]))
df1.stemmed.head()


# In[29]:


cvec1 = CountVectorizer(stop_words= stops, min_df=0.001, max_df=0.9, ngram_range=(1,2))


# In[30]:


#from itertools import islice
cvec1.fit(df1.stemmed)
list(islice(cvec1.vocabulary_.items(), 20))


# In[31]:


cvec1 = CountVectorizer(stop_words=stops, min_df=.001, max_df=.8, ngram_range=(1,2))
cvec1.fit(df1.stemmed)
len(cvec1.vocabulary_)


# In[32]:


cvec_counts1 = cvec.transform(df1.stemmed)
print ('sparse matrix shape:', cvec_counts1.shape)
print ('nonzero count:', cvec_counts1.nnz)
print ('sparsity: %.2f%%' % (100.0 * cvec_counts1.nnz / (cvec_counts1.shape[0] * cvec_counts1.shape[1])))


# In[33]:


transformer1 = TfidfTransformer()
transformed_weights1 = transformer.fit_transform(cvec_counts1)
transformed_weights1


# In[34]:


preds = nb.predict(transformed_weights1)


# In[35]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[36]:


confusion_matrix(df1["Class"], preds)


# In[37]:


accuracy_score(df1["Class"], preds, normalize=True)

