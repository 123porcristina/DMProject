# import pandas as pd
# import nltk
# from nltk import word_tokenize, sent_tokenize
# from nltk.stem import PorterStemmer, LancasterStemmer
# from nltk.corpus import stopwords
# from nltk.probability import FreqDist
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn import metrics
# from textblob import TextBlob, Word
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re, string, unicodedata
# from sklearn.svm import SVC

# #This is the main file.
# print("this is main file")
# import numpy as np
# import pandas as pd
#
# df = pd.read_csv("Datafiniti_Hotel_Reviews.csv")
# df["reviews.text"] = df["reviews.text"].astype(str)
# print(df.head())
# print(df.dtypes)
# print(df["reviews.text"].isna().sum())
# print(df["reviews.title"].notnull().isna().sum())
#
# #Basic feature extraction using text data
#
# # Number of Words (the negative sentiments contain a lesser amount of words than the positive ones.)
# df["wordcount_reviews.text"]=df["reviews.text"].apply(lambda x: len(str(x).split(" ")))
# df["wordcount_reviews.title"]=df["reviews.title"].apply(lambda x: len(str(x).split(" ")))
#
# # Number of characters (includes spaces)
# df["charcount_reviews.text"] = df["reviews.text"].str.len()
# df["charcount_reviews.title"] = df["reviews.title"].str.len()
#
# # Average Word Length
# def avg_word(reviews):
#   words = str(reviews).split()
#   return (sum(len(word) for word in words)/len(words))
#
# df["avgword_reviews.text"] = df["reviews.text"].apply(lambda x: avg_word(x))
# df["avgword_reviews.title"] = df["reviews.title"].apply(lambda x: avg_word(x))
#
# # Number of stopwords, need to remove the stop word, but need to how many of them
# import nltk
# from nltk.corpus import stopwords
# stop=stopwords.words('english')
# df["stopwords_reviews.text"] = df["reviews.text"].apply(lambda x: len([x for x in str(x).split() if x in stop]))
# df["stopwords_reviews.title"] = df["reviews.title"].notnull().apply(lambda x: len([x for x in str(x).split() if x in stop]))
#
# # Number of numerics
# df["numerics_reviews.text"] = df["reviews.text"].apply(lambda x: len([x for x in str(x).split() if x.isdigit()]))
# df["numerics_reviews.title"] = df["reviews.title"].apply(lambda x: len([x for x in str(x).split() if x.isdigit()]))
#
# # Number of Uppercase words (Anger or rage is quite often expressed by writing in UPPERCASE words )
# df['upper_reviews.text'] = df['reviews.text'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))
# df['upper_reviews.title'] = df['reviews.title'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))
#
# print(df[["reviews.text","wordcount_reviews.text","charcount_reviews.text","avgword_reviews.text","stopwords_reviews.text","numerics_reviews.text",'upper_reviews.text']].head())
# print(df[["reviews.title","wordcount_reviews.title","charcount_reviews.title","avgword_reviews.title","stopwords_reviews.title","numerics_reviews.title",'upper_reviews.title']].head())

# Pre-processing
#e
# Lowr case
# df["reviews.text"] = df['reviews.text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# print(df['reviews.text'].head())
#
# #Remove Punctuation
# df["reviews.text"] = df["reviews.text"].str.replace('[^\w\s]',"")
# print(df['reviews.text'].head())
#
# #Removal of Stop Words
# stop = stopwords.words('english')
# df["reviews.text"] = df["reviews.text"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# print(df["reviews.text"].head())
#
# #check the top ten common words, looks some of the common words may useful, we decide to retain for now.
# freq = pd.Series(" ".join(df["reviews.text"]).split()).value_counts()[:10]
# print(freq)
#
# #check the rare words, looks some of these words may cause by wrong spelling, so we decide to do the spelling correction
# freq = pd.Series(" ".join(df["reviews.text"]).split()).value_counts()[-10:]
# print(freq)
# Kevin. SVM with rbf, including the compare with scale
# def svm_apply(self):
#     X_train, X_test, y_train, y_test = train_test_split(self.f_df['reviews_text'],
#                                                         self.f_df['reviews_rating'].astype('int'), test_size=0.25,
#                                                         random_state=85)
#     count_vectorizer = CountVectorizer()
#     count_train = count_vectorizer.fit_transform(X_train.values)
#     count_test = count_vectorizer.transform(X_test.values)
#     SVM = SVC(C=10, kernel='rbf', gamma='auto')
#     SVM.fit(count_train, y_train)
#     predictions_SVM = SVM.predict(count_test)
#     return "Accuracy score SVM without scale:" + str(metrics.accuracy_score(y_test, predictions_SVM))

#     def count_chr(self):
#         # Number of characters (includes spaces)
#         self.pr_df["charcount_reviews.text"] = self.pr_df["reviews_text"].str.len()
#         self.pr_df["charcount_reviews.title"] = self.pr_df["reviews_title"].str.len()
#
#     # Kevin
#     def rating_negative_Moderate_positive(self):
#         #Kevin. scale convert rating into negative(x >= 0,x < 3),moderate(x >= 3 and x <= 4), positive(x<4 and x<5)
#         self.pr_df["New_reviews_rating_NMP"] = self.pr_df["reviews_rating"].apply(
#             lambda x: 1 if x > 0 and x < 3 else (2 if x >= 3  and x < 4 else 3))
#         return self.pr_df
#
# def svm_apply_1(self):
#     X_train, X_test, y_train, y_test = train_test_split(self.f_df['reviews_text'],
#                                                         self.f_df["New_reviews_rating_NMP"].astype('int'),
#                                                         test_size=0.25, random_state=85)
#     count_vectorizer = CountVectorizer()
#     count_train = count_vectorizer.fit_transform(X_train.values)
#     count_test = count_vectorizer.transform(X_test.values)
#     SVM = SVC(C=1.0, kernel='rbf', gamma='auto')
#     SVM.fit(count_train, y_train)
#     predictions_SVM = SVM.predict(count_test)
#     return "Accuracy score SVM with NMP Scale: " + str(metrics.accuracy_score(y_test, predictions_SVM))