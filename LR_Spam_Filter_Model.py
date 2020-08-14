#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#importing dataset
columns=['category','text']
data=pd.read_csv("C:/Users/user/Downloads/smsspamcollection/SMSSpamCollection.txt",names=columns,sep='[\t]',header=None,engine='python')


# In[4]:


print(data)
print(data.shape)


# In[5]:


data['bool_cat']=data['category'].map({'ham':0,'spam':1})
print(data)
data['Count']=0
#counting the length of each sms
for i in np.arange(0,len(data['text'])):
    data.loc[i,'Count']=len(data.loc[i,'text'])
data


# In[6]:


#storing spam and sorting based on index
spam=data[data['bool_cat']==1]
print(spam,spam.shape)
spam_count=pd.DataFrame(pd.value_counts(spam['Count']).sort_index())
spam_count


# In[7]:


#storing ham and sorting based on index
ham=data[data['bool_cat']==0]
print(ham,ham.shape)
ham_count=pd.DataFrame(pd.value_counts(ham['Count']).sort_index())
ham_count


# In[8]:


#plotting the spam and ham on subplot to get idea about the word count of spam messages
fig, ax = plt.subplots(figsize=(17,5))
spam_count['Count'].sort_index().plot(ax=ax, kind='bar',facecolor='red');
ham_count['Count'].sort_index().plot(ax=ax, kind='bar',facecolor='green');


# In[9]:


import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
porter=PorterStemmer()
lancaster=LancasterStemmer()
def stem_txt(text):
    token_words=word_tokenize(text)
    stem_text=[]
    for word in token_words:
        word=str.lower(word)
        if word not in stopwords.words():
            stem_text.append(porter.stem(word))
            stem_text.append(" ")
    return "".join(stem_text)
#spam['stemmed_text']=[]
for i in spam.index:
    text=spam.loc[i,'text']
    spam.loc[i,'stemmed_ext']=stem_txt(text)
spam


# In[20]:


#lematizing the sms after removing stopwords from text and tokenizing
from nltk.stem import WordNetLemmatizer#nltk means natural language toolkit
wordnet_lemmatizer=WordNetLemmatizer()#wordnet lemmatizer has been used
#function to lemmatize
def lemmatize_txt(text):
    token_words=word_tokenize(text)#tokenizing the text
    lemmatized_text=[]
    for word in token_words:
        word=str.lower(word) #converting all text to lowercase
        if not word in stopwords.words():  #removingg stopwords
            lemmatized_text.append(wordnet_lemmatizer.lemmatize(word,pos='v'))
            lemmatized_text.append(" ")
    return "".join(lemmatized_text) #again join lemmatized tokens to form sentences

for i in data.index:
    text=data.loc[i,'text']
    data.loc[i,'lemmatized_text']=lemmatize_txt(text) #passing the sms text for lemmatization
data


# In[23]:


#vectorizing the words in sms text for feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#tf idf vectorizer function is used here
Vectorizer=TfidfVectorizer() 
X=Vectorizer.fit_transform(data.lemmatized_text)
print(X)
Y=data.bool_cat
Y


# In[28]:


#splitting the overall data into training setand test set based on a ratio randomly
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.25,train_size=.85,random_state=42)
print("Training set has {} samples".format(X_train.shape[0]))
print("Testing set has {} samples".format(X_test.shape[0]))


# In[57]:


#building the model
from sklearn.linear_model import LogisticRegression
LogReg=LogisticRegression(solver='lbfgs')
#lbfgs is LR solver out of many other solvers but it works for multnomial option also so
#fitting the model
LogReg.fit(X_train,Y_train)


# In[62]:


#predict dependent variables for training and test data
Y_predict=LogReg.predict(X_test)
Y_predict_train=LogReg.predict(X_train)
print(Y_predict)
print(Y_predict_train)
y_prob_train=LogReg.predict_proba(X_train)[:,1].reshape(1,-1)
print(y_prob_train)
y_prob=LogReg.predict_proba(X_test)[:,1]
y_prob


# In[56]:


#results of performance
from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test,Y_predict)#checking accuracy
print(score)
from sklearn.metrics import confusion_matrix
#confusion matrix
conf_matrix=confusion_matrix(Y_test,Y_predict)
print(conf_matrix)
tn,fp,fn,tp=conf_matrix[0][0],conf_matrix[0][1],conf_matrix[1][0],conf_matrix[1][1]
accuracy=(tn+tp)/(tn+tp+fp+fn)#tn true negative, fn false negative, tp true positive, fp false positive
print("accuracy {:0.2f}".format(accuracy))
sensitivity=tp/(tp+fn) 
#sensitivity is measure of probabability of correctly identifying spam
print("sensitivity {:0.2f}".format(sensitivity))
specificity=tn/(tn+fp)
#specificity is measure of probability of correctly identifying ham
print("specificity {:0.2f}".format(specificity))

