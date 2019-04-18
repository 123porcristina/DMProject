import pandas as pd
import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

#this is uploaded from App.py as df
df = pd.read_csv('datafiniti_hotel_reviews.csv')


def missing_val(): #Identifing missing values in the dataframe

    df.isnull().values.any()
    df["reviews_text"].isna().sum() #Kevin
    df["reviews_title"].notnull().isna().sum() #Kevin

def drop_columns(): #Delete columns that are not useful for our dataset
                    #This is our feature selection - Reduce dimension

    dropcols = ['dateupdated', 'address', 'categories', 'keys',
                'dateadded',  'reviews_dateseen', 'reviews_sourceurls',
                'websites', 'location', 'reviews_username']

    return df.drop(dropcols)


def noise_removal():


def remove_stop_words():


def train():
  train_df = pandas.DataFrame()
  train_df['text'] = texts
  train_df['label'] = labels

def cont_neg_feel(): # Number of Words (the negative sentiments contain
                     # a lesser amount of words than the positive ones.)
  df["wordcount_reviews.text"]=df["reviews.text"].apply(lambda x: len(str(x).split(" ")))
  df["wordcount_reviews.title"]=df["reviews.title"].apply(lambda x: len(str(x).split(" ")))


def count_chr(): # Number of characters (includes spaces)
  df["charcount_reviews.text"] = df["reviews.text"].str.len()
  df["charcount_reviews.title"] = df["reviews.title"].str.len()



# #df = pd.read_csv("Datafiniti_Hotel_Reviews.csv")
# # df["reviews.text"] = df["reviews.text"].astype(str)
# # print(df.head())
# # print(df.dtypes)
# # print(df["reviews.text"].isna().sum())
# # print(df["reviews.title"].notnull().isna().sum())
# #
# # #Basic feature extraction using text data
# #
# # # Number of Words (the negative sentiments contain a lesser amount of words than the positive ones.)
# # df["wordcount_reviews.text"]=df["reviews.text"].apply(lambda x: len(str(x).split(" ")))
# # df["wordcount_reviews.title"]=df["reviews.title"].apply(lambda x: len(str(x).split(" ")))
# #
# # # Number of characters (includes spaces)
# # df["charcount_reviews.text"] = df["reviews.text"].str.len()
# # df["charcount_reviews.title"] = df["reviews.title"].str.len()
# #
# # # Average Word Length
# # def avg_word(reviews):
# #   words = str(reviews).split()
# #   return (sum(len(word) for word in words)/len(words))
# #
# # df["avgword_reviews.text"] = df["reviews.text"].apply(lambda x: avg_word(x))
# # df["avgword_reviews.title"] = df["reviews.title"].apply(lambda x: avg_word(x))
# #
# # # Number of stopwords, need to remove the stop word, but need to how many of them
# # import nltk
# # from nltk.corpus import stopwords
# # stop=stopwords.words('english')
# # df["stopwords_reviews.text"] = df["reviews.text"].apply(lambda x: len([x for x in str(x).split() if x in stop]))
# # df["stopwords_reviews.title"] = df["reviews.title"].notnull().apply(lambda x: len([x for x in str(x).split() if x in stop]))
# #
# # # Number of numerics
# # df["numerics_reviews.text"] = df["reviews.text"].apply(lambda x: len([x for x in str(x).split() if x.isdigit()]))
# # df["numerics_reviews.title"] = df["reviews.title"].apply(lambda x: len([x for x in str(x).split() if x.isdigit()]))
# #
# # # Number of Uppercase words (Anger or rage is quite often expressed by writing in UPPERCASE words )
# # df['upper_reviews.text'] = df['reviews.text'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))
# # df['upper_reviews.title'] = df['reviews.title'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))
# #
# # print(df[["reviews.text","wordcount_reviews.text","charcount_reviews.text","avgword_reviews.text","stopwords_reviews.text","numerics_reviews.text",'upper_reviews.text']].head())
# # print(df[["reviews.title","wordcount_reviews.title","charcount_reviews.title","avgword_reviews.title","stopwords_reviews.title","numerics_reviews.title",'upper_reviews.title']].head())
# #
# #
# # # Pre-processing
# #
# # #Lower case
# # df["reviews.text"] = df['reviews.text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# # print(df['reviews.text'].head())
# #
# # #Remove Punctuation
# # df["reviews.text"] = df["reviews.text"].str.replace('[^\w\s]',"")
# # print(df['reviews.text'].head())
# #
# # #Removal of Stop Words
# # stop = stopwords.words('english')
# # df["reviews.text"] = df["reviews.text"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# # print(df["reviews.text"].head())
# #
# # #check the top ten common words, looks some of the common words may useful, we decide to retain for now.
# # freq = pd.Series(" ".join(df["reviews.text"]).split()).value_counts()[:10]
# # print(freq)
# #
# # #check the rare words, looks some of these words may cause by wrong spelling, so we decide to do the spelling correction
# # freq = pd.Series(" ".join(df["reviews.text"]).split()).value_counts()[-10:]
# # print(freq)