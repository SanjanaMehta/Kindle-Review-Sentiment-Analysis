#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\Sanjana\Downloads\my documents\projects\kindle\all_kindle_review .csv")


# In[3]:


cols=df[['Unnamed: 0.1','Unnamed: 0']]
df=df.drop(cols,axis=1)


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df=df[['reviewText','rating']]
df.info()


# In[7]:


df.head()


# In[8]:


df['rating'].value_counts()


# In[9]:


df['rating']=df['rating'].apply(lambda x:0 if x<3 else 1)


# ### Text cleaning

# In[10]:


df['reviewText']=df['reviewText'].str.lower()


# In[11]:


import re
df['reviewText']=df['reviewText'].apply(lambda x:re.sub('[^a-z A-z 0-9]+',' ',x))


# In[12]:


import nltk
nltk.download('stopwords')


# In[13]:


from nltk.corpus import stopwords
df['reviewText']=df['reviewText'].apply(lambda x: " ".join([y for y in x.split() if y not in stopwords.words('english')]))


# In[14]:


df['reviewText']=df['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^%&:/~+#-]*[\w@?^=%&/~+#-])?',' ',str(x)))


# In[15]:


from bs4 import BeautifulSoup
df['reviewText']=df['reviewText'].apply(lambda x: BeautifulSoup(x,'lxml').get_text())


# In[16]:


df['reviewText']=df['reviewText'].apply(lambda x:" ".join(x.split()))


# In[17]:


df.head()


# In[18]:


from nltk.stem import WordNetLemmatizer
lemmatize=WordNetLemmatizer()


# In[21]:


def lemmatize_words(text):
    return " ".join([lemmatize.lemmatize(word) for word in text.split()])


# In[22]:


df['reviewText']=df['reviewText'].apply(lambda x:lemmatize_words(x))


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(df['reviewText'],df['rating'],test_size=0.25, random_state=42)


# In[27]:


X_train.shape


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer
bow=CountVectorizer()


# In[32]:


X_train_bow=bow.fit_transform(X_train).toarray()
X_test_bow=bow.transform(X_test).toarray()


# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()


# In[34]:


X_train_tf=tfidf.fit_transform(X_train).toarray()
X_test_td=tfidf.transform(X_test).toarray()


# In[36]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB().fit(X_train_bow,y_train)


# In[39]:


nb_tfidf=GaussianNB().fit(X_train_tf,y_train)


# In[40]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[52]:


y_pred_bow=nb.predict(X_test_bow)


# In[53]:


y_test.shape,y_pred_bow.shape


# In[55]:


y_pred_tfidf=nb_tfidf.predict(X_test_td)


# In[56]:


print(accuracy_score(y_test,y_pred_bow))


# In[57]:


print(confusion_matrix(y_test,y_pred_bow))
print(classification_report(y_test,y_pred_bow))


# In[59]:


print(accuracy_score(y_test,y_pred_tfidf))
print(confusion_matrix(y_test,y_pred_tfidf))
print(classification_report(y_test,y_pred_tfidf))


# In[ ]:




