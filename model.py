#!/usr/bin/env python
# coding: utf-8

# # Case Study
# ## Sentiment Based Product Recommendation System
# **Author : Sarika Srivastava**

# In[2]:


#importing relevant libraries
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns


# In[100]:


#import nltk libraries for NLP
import nltk
from nltk.corpus import stopwords
from nltk  import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize


# In[161]:


# To show all the columns
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 300)

# Avoid warnings
import warnings
warnings.filterwarnings("ignore")


# In[232]:


#import libraries for model evaluation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import KFold


# In[3]:


#set the print options to view dataset properly

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format # to display upto 2 decimal places


# In[126]:


data = pd.read_csv("sample30.csv")


# In[127]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


# ## Data Cleaning

# In[128]:


round(data.isna().sum()/len(data.index)*100,2)


# **We can see there are 2 columns with missing values more than 90% , City name and Province name of the user. Since we do not need them so we can drop them**

# In[130]:


drop_column = ['reviews_userCity','reviews_userProvince']


# **There are 2 columns reviews_didPurchase and reviews_doRecommend which are of no use so we can drop them**

# In[131]:


drop_column.append('reviews_didPurchase')
drop_column.append('reviews_doRecommend')


# In[132]:


#drop columns with the missing value more than 90%
data.drop(columns =drop_column,inplace = True)


# In[133]:


#check the missing value again
round(data.isna().sum()/len(data.index)*100,2)


# **Rest of the columns with missing values, we will drop those rows**

# In[135]:


data.dropna(subset = ['manufacturer','reviews_date','reviews_title','reviews_username'],inplace = True)


# In[138]:


#check the missing value again
round(data.isna().sum()/len(data.index)*100,2)


# In[136]:


data.shape


# In[137]:


#check number of unique values in each column
for i in data.columns:
    print(i,': ',data[i].nunique())


# In[139]:


data.columns


# **user_sentiment is our target variable. This is a binaru variable so we will convert it into numerical form**

# In[141]:


#Positive to 1 and Negative to 0
data['user_sentiment']=data['user_sentiment'].apply(lambda x: 1 if x== 'Positive' else 0)


# In[143]:


data.head(2)


# ### EDA

# In[80]:


# we will check the rating distribution
sns.countplot(data=data, x='reviews_rating')


# **We can see above that many users have given a rating of 5 to products followed by 4 and 3 whereas very few users have given a low rating of 1 or 2.**

# In[78]:


# lets check the genuine number of reviews i.e. the reviews for which product was actually purchased
plt.figure()
ax = sns.countplot(data['reviews_didPurchase'])
ax.set_xlabel(xlabel="Did Purchase", fontsize=12)
ax.set_ylabel(ylabel='No. of Reviews', fontsize=12)
ax.axes.set_title('Genuine No. of Reviews', fontsize=12)
ax.tick_params(labelsize=13)


# **We can see very less people has purchased the item and reviewed also.**

# In[81]:


sns.countplot(data=data, x='user_sentiment')


# We can see most of the users have given positive reviews

# **Lets check which are the top 10 most rated products**

# In[114]:


data[data['reviews_didPurchase'] == True]['name'].value_counts()[0:10]


# In[111]:


data[data['reviews_didPurchase'] == True]['name'].value_counts()[0:10].plot(kind = 'barh').invert_yaxis()


# In[116]:


df = data[data['name']=="Hormel Chili, No Beans"]
sns.countplot(data=df, x='reviews_rating')


# **Target value distribution**

# In[165]:


sns.countplot(data=data, x='user_sentiment')


# **We can see the data is highly imbalanced**
# **We will perform some sampling to reslove this issue later during ML algorithm**

# ## Text preprocessing

# **As the review is mostly text data, we might need to preprocess the data to gain some useful insights from the data**

# **For prediction, we dont need all columns, we need reviews title, review text and user_sentiment.**
# **We can either keep review_title and review_text seperately or we can combine both**
# **I will combine the two columns**

# In[146]:


# Joining Review Text and Title.
data['Review'] = data['reviews_title'] + " " + data['reviews_text'] 


# In[147]:


#lets have a look at few random reviews
for i in range(0,30,5):
    print(data['Review'][i])
    print("**********")


# In[148]:


# Clean the reviews    
data['cleaned_text'] = data['Review'].replace(r'\'|\"|\,|\.|\?|\+|\-|\/|\=|\(|\)|\n|"', '', regex=True)
data['cleaned_text'] = data['cleaned_text'].replace("  ", " ")


# In[149]:


words_remove = ["ax","i","you","edu","s","t","m","subject","can","lines","re","what", "there","all","we",
                "one","the","a","an","of","or","in","for","by","on","but","is","in","a","not","with","as",
                "was","if","they","are","this","and","it","have","has","from","at","my","be","by","not","that","to",
                "from","com","org","like","likes","so","said","from","what","told","over","more","other",
                "have","last","with","this","that","such","when","been","says","will","also","where","why",
                "would","today", "in", "on", "you", "r", "d", "u", "hw","wat", "oly", "s", "b", "ht", 
                "rt", "p","the","th", "n", "was"]


def cleantext(data, words_to_remove = words_remove): 
    # remove emoticons form the reviews if any
    data['cleaned_text'] = data['cleaned_text'].replace(r'<ed>','', regex = True)
    data['cleaned_text'] = data['cleaned_text'].replace(r'\B<U+.*>|<U+.*>\B|<U+.*>','', regex = True)
    
    # convert reviews to lowercase
    data['cleaned_text'] = data['cleaned_text'].str.lower()
            
    #remove_symbols if any
    data['cleaned_text'] = data['cleaned_text'].replace(r'[^a-zA-Z0-9]', " ", regex=True)
    
    #remove punctuations 
    data['cleaned_text'] = data['cleaned_text'].replace(r'[[]!"#$%\'()\*+,-./:;<=>?^_`{|}]+',"", regex = True)

    
    #remove words of length 1 or 2 
    data['cleaned_text'] = data['cleaned_text'].replace(r'\b[a-zA-Z]{1,2}\b','', regex=True)

    #remove extra spaces in the review
    data['cleaned_text'] = data['cleaned_text'].replace(r'^\s+|\s+$'," ", regex=True)
     
    #remove_digits
    data['cleaned_text'] = data['cleaned_text'].replace(r'[0-9]', "", regex=True)
    
    #remove stopwords and words_to_remove
    stop_words = set(stopwords.words('english'))
    mystopwords = [stop_words, "via", words_to_remove]
    
    data['fully_cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in mystopwords]))
    
    return data

#get the processed reviews
data = cleantext(data)


# In[150]:


data.columns


# In[151]:


#lets have a look at few random cleaned reviews
for i in range(0,30,5):
    print(data['fully_cleaned_text'][i])
    print("**********")


# **Now will perform below operations over the review text :**<br>
# **1. Tokenization**<br>
# **2. Normalizing Words(Lemma or Stemming)**<br>

# In[152]:


data['review_token'] = data['fully_cleaned_text'].apply(word_tokenize)
data.head(2)


# **We will use lemmatization, but we cannot do this without POS tagging, so created a function to return the lemmetizor tags and then return the lemma of the word list**

# In[153]:


lemmatizer = WordNetLemmatizer()


# In[154]:


def get_wordnet_pos(treebank_tag):
    #return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
            # As default pos in lemmatization is Noun
        return wordnet.NOUN

def pos_tag_1(tokens):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
    lemma = []
    for word, tag in pos_tag(tokens):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma.append(word)
        else:
            lemma.append(lemmatizer.lemmatize(word, wntag))
    return lemma


# In[155]:


data['text_lemmatized'] = data.review_token.apply(lambda x: pos_tag_1(x))


# In[156]:


data[['review_token','text_lemmatized']]


# In[157]:


#At last we will join all the lemmatized words to create a final review for furthur process
data['final_review'] = data['text_lemmatized'].apply(lambda x: ' '.join(word for word in x))


# In[162]:


data['final_review'].head()


# **Before Feature Extraction, we will split the data into train and test**

# In[163]:


data.columns


# ### Feature Extraction - Tfidf

# In[164]:


x=data['final_review'] 
y=data['user_sentiment']


# In[166]:


# Split the dataset into test and train
seed = 50 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# In[ ]:


word_vectorizer = TfidfVectorizer(
    strip_accents='unicode',     
    analyzer='word',            # ngrams will be of words
    token_pattern=r'\w{1,}',    
    ngram_range=(1, 3),         
    stop_words='english',
    sublinear_tf=True)


# In[169]:


# we will fit_transform X_train and only transform X_test
X_train_tfidf = word_vectorizer.fit_transform(X_train)
X_test_tfidf = word_vectorizer.transform(X_test)


# In[170]:


# Print the shape of train and test dataset
print('X_train', X_train_tfidf.shape)
print('y_train', y_train.shape)
print('X_test', X_test_tfidf.shape)
print('y_test', y_test.shape)


# In[196]:


#create a function for plotting confusion matrix

def cm_plot(cm_train,cm_test):
    
    print("Confusion matrix for train and test data set")

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%',cmap="PiYG")
    

    plt.subplot(1,2,2)
    sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%',cmap="PiYG")

    plt.show()


# In[206]:


#create a function for calculating Sensitivity and Specificity
def spec_sensitivity(cm_train,cm_test):
    
    #Train
    tn, fp, fn, tp = cm_train.ravel()
    specificity_train = tn / (tn+fp)
    sensitivity_train = tp / float(fn + tp)
    
    print("sensitivity for train set: ",sensitivity_train)
    print("specificity for train set: ",specificity_train)
    print("\n****\n")
    
    #Test
    tn, fp, fn, tp = cm_test.ravel()
    specificity_test = tn / (tn+fp)
    sensitivity_test = tp / float(fn + tp)
    
    print("sensitivity for test set: ",sensitivity_test)
    print("specificity for train set: ",specificity_test)


# ### Training a text classification model

# **Before training the text classification, we will do some sampling because in EDA we saw the target data is highly imbalanced**<br>**There are two techniques for sampling :**<br>
# **1. Oversampling**<br>
# **2. SMOTE**

# In[171]:


#from imblearn import over_sampling
from imblearn import over_sampling
over = over_sampling.RandomOverSampler(random_state=0)


# In[172]:


# Oversampling the dataset.
X_train, y_train = over.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))


# In[173]:


pd.Series(y_train).value_counts()


# In[174]:


# we convert the 2D array to a dataframe and then convert it to a list.
X_train = pd.DataFrame(X_train).iloc[:,0].tolist()


# In[175]:


# transforming the train and test datasets

X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())


# ## Logistic Regression

# In[176]:


# Building Logistic Regression
t1 = time.time()

cls_log = LogisticRegression()
cls_log.fit(X_train_transformed,y_train)

t2 = time.time()
print('Time Taken: {:.2f} seconds'.format(t2-t1))


# **Model Accuracy on train data**

# In[181]:


y_train_pred_logit = cls_log.predict(X_train_transformed)

print("Logistic Regression accuracy on train data", accuracy_score(y_train_pred_logit, y_train),"\n")
print(classification_report(y_train_pred_logit, y_train))


# **Model Accuracy on test data**

# In[182]:


y_test_pred_logit = cls_log.predict(X_test_transformed)

print("Logistic Regression accuracy on test data", accuracy_score(y_test_pred_logit, y_test),"\n")
print(classification_report(y_test_pred_logit, y_test))


# **We can see the Maxro AVg is 77%**

# In[ ]:


cm_train = metrics.confusion_matrix(y_train, y_train_pred_logit)
cm_test = metrics.confusion_matrix(y_test, y_test_pred_logit)


# In[205]:


spec_sensitivity(cm_train,cm_test)


# In[198]:


cm_plot(cm_train,cm_test)


# # Recommendation System

# **Now we will move to Recommendation system.**<br>**For this we will first recognise relevant columns out of all columns**

# In[241]:


# recognize relevant columns
data.columns


# **We need below columns**<br>
# **1. reviews_username -- The unique identification for individual user in the dataset**<br>
# **2. reviews_rating -- Rating given by the user to a particular product**<br>
# **3. name -- Name of the product to which user has added review or rating**

# In[242]:


#we will create a new dataframe with above columns

data_recom = data[['reviews_username','reviews_rating','name']]


# In[243]:


data_recom.head(5)


# **We will divide the dataset into train and test**

# In[244]:


train, test = train_test_split(data_recom, test_size=0.30, random_state=31)


# **Creating dummy train & dummy test dataset**

# These dataset will be used for prediction 
# - Dummy train will be used later for prediction of the product which has not been rated by the user. To ignore the product rated by the user, we will mark it as 0 during prediction. The products not rated by user is marked as 1 for prediction in dummy train dataset. 
# 
# - Dummy test will be used for evaluation. To evaluate, we will only make prediction on the products rated by the user. So, this is marked as 1. This is just opposite of dummy_train.

# In[245]:


# Copy the train dataset into dummy_train
dummy_train = train.copy()


# In[246]:


# The products not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[247]:


dummy_train.head()


# In[249]:


# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(1)


# In[250]:


dummy_train.head()


# ### User Similarity Matrix¶

# **Cosine Similarity**
# 
# Cosine Similarity is a measurement that quantifies the similarity between two vectors [Which is Rating Vector in this case] 
# 
# **Adjusted Cosine**
# 
# Adjusted cosine similarity is a modified version of vector-based similarity where we incorporate the fact that different users have different ratings schemes. In other words, some users might rate items highly in general, and others might give items lower ratings as a preference. To handle this nature from rating given by user , we subtract average ratings for each user from each user's rating for different movies.
# 

# In[251]:


# Create a user-movie matrix.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
)


# In[252]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[253]:


df_subtracted.head()


# In[254]:


from sklearn.metrics.pairwise import pairwise_distances


# In[255]:


# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[256]:


np.shape(user_correlation)


# ### Prediction - User User

# Doing the prediction for the users which are positively related with other users, and not the users which are negatively related as we are interested in the users which are more similar to the current users. So, ignoring the correlation for values less than 0.

# In[257]:


user_correlation[user_correlation<0]=0
user_correlation


# Rating predicted by the user (for movies rated as well as not rated) is the weighted sum of correlation with the movie rating (as present in the rating dataset).

# In[258]:


user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings


# In[259]:


user_predicted_ratings.shape


# Since we are interested only in the products not rated by the user, we will ignore the products rated by the user by making it zero.

# In[260]:


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# ### Evaluation - User User

# Evaluation will be same as we have seen above for the prediction. The only difference being, we will evaluate for the product already rated by the user insead of predicting it for the product not rated by the user.

# In[263]:


# Find out the common users of test and train dataset.
common = test[test.reviews_username.isin(train.reviews_username)]
common.shape


# In[264]:


common.head()


# In[265]:


# convert into the user-product matrix.
common_user_based_matrix = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


# In[266]:


# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)


# In[267]:


df_subtracted.head(1)


# In[268]:


user_correlation_df['reviews_username'] = df_subtracted.index
user_correlation_df.set_index('reviews_username',inplace=True)
user_correlation_df.head()


# In[269]:


list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_subtracted.index.tolist()


user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]


# In[270]:


user_correlation_df_1.shape


# In[271]:


user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]


# In[272]:


user_correlation_df_3 = user_correlation_df_2.T


# In[273]:


user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings


# In[274]:


dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)


# In[275]:


dummy_test.shape


# In[276]:


common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)


# In[277]:


common_user_predicted_ratings.head(2)


# Calculating the RMSE for only the products rated by user. For RMSE, normalising the rating to (1,5) range.

# In[278]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[279]:


common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


# In[280]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[281]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# In[314]:


# save the respective files Pickle 
import pickle
pickle.dump(user_final_rating,open('user_final_rating.pkl','wb'))
user_final_rating =  pickle.load(open('user_final_rating.pkl', 'rb'))


# ## Recommendation

# In[318]:


# Take the user ID as input
user_input = input("Enter your user name")
print(user_input)


# ### Finding the top 20 recommendation for the user

# In[319]:


# Recommending the Top 5 products to the user.
d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
d


# **Filtering out the Top 5 recommendation items based on Logistic Regression ML model.**

# In[320]:


# save the respective files and models through Pickle 
import pickle
pickle.dump(cls_log,open('logit_model.pkl', 'wb'))
# loading pickle object
cls_log =  pickle.load(open('logit_model.pkl', 'rb'))

pickle.dump(word_vectorizer,open('word_vectorizer.pkl','wb'))
# loading pickle object
word_vectorizer = pickle.load(open('word_vectorizer.pkl','rb'))


# In[325]:


# Define a function to recommend top 5 filtered products to the user.
def recommend(user_input):
    d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]

    # Based on positive sentiment percentage.
    i= 0
    a = {}
    for prod_name in d.index.tolist():
        product = prod_name
        product_name_review_list =data[data['name']== product]['final_review'].tolist()
        features= word_vectorizer.transform(product_name_review_list)
        cls_log.predict(features)
        a[product] = cls_log.predict(features).mean()*100
    b= pd.Series(a).sort_values(ascending = False).head(5).index.tolist()
    print("Username : ",user_input,'\n**************************\n')
    print("Top 5 product recommendation you may like :")
    for i,val in enumerate(b):
        print(i+1,val)


# In[326]:


recommend(user_input)


# In[ ]:


data.to_csv("data.csv",index=False)

