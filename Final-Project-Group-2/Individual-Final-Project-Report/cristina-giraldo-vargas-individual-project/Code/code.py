import base64
import io
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_table
import pandas as pd
import CleaningProcess as dmproject
import dash_core_components as dcc
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([

        html.Div(children=[
            html.H2(children='Text Analysis Hotel Reviews in the U.S',
                    style={'textAlign': 'center', 'color': '#0f2128'},
                    className="twelve columns"),  # title occupies 9 cols

            html.Div(children=''' 
                        Dash: Text Analysis Hotel Reviews in the U.S.
                        ''',
                     className="nine columns")  # this subtitle occupies 9 columns
        ], className="row"),

        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div([
                html.Div(id='output-data-upload'),
                # html.Div(id='output-data-info'),
            ], className="row")
        ], className='row'),
    ])
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            clean = dmproject.CleaningDF(df)
            df = clean.drop_columns()  # Cristina. Drop features that are not necessary for the analysis
            df = clean.missing_val()  # Cristina - Kevin. Verify and clean missing values and converts to string reviews_text

            # 3. PREPROCESSING
            clean_text = dmproject.PreprocessReview(df)
            freq = clean_text.common_words(df['reviews_text'], 25)
            df = clean_text.clean_split_text()  # Cristina - Kevin. Converts to lower case, removes punctuation.
            df = clean_text.remove_stop_w()  # Renzo. It removes stop words from reviews_text
            cwc = clean_text.count_rare_word()  # Renzo. Count the rare words in the reviews. I tried with :10, then with :-20
            df = clean_text.remove_rare_words()  # Renzo. This will clean the rare words from the reviews column
            freq_A = clean_text.common_words(df['reviews_text'],
                                             25)  # Cristina. Shows the frequency of stop words AFTER removing

            words = freq['word'].tolist()
            twords = freq['count'].tolist()

            words_A = freq_A['word'].tolist()
            twords_A = freq_A['count'].tolist()


        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([

        html.Div([

            html.Div([
                dash_table.DataTable(
                    data=df.to_dict('rows'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    style_table={'overflowX': 'scroll', 'maxHeight': '400px', 'overflowY': 'scroll'},
                    style_cell={'padding': '5px',
                                'whiteSpace': 'no-wrap',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'maxWidth': 0,
                                'height': 30,
                                'textAlign': 'left'},
                    style_header={'backgroundColor': 'white',
                                  'fontWeight': 'bold',
                                  'color': 'black'}
                ),
            ], className="seven columns"),

            html.Div([

                html.Div([
                    dcc.Graph(
                        figure=go.Figure(
                            data=[

                                go.Bar(
                                    x=words,
                                    y=twords,
                                    name='words Before cleaning',
                                    marker=go.bar.Marker(
                                        color='rgb(55, 83, 109)'
                                    )
                                )
                            ],
                            layout=go.Layout(
                                title='BEFORE PREPROCESSING',
                                showlegend=True,
                                legend=go.layout.Legend(
                                    x=0,
                                    y=1.0
                                ),
                                margin=go.layout.Margin(l=40, r=0, t=40, b=30)
                            )
                        )
                    ),
                ], className='six columns'),

                html.Div([
                    dcc.Graph(
                        figure=go.Figure(
                            data=[

                                go.Bar(
                                    x=words_A,
                                    y=twords_A,
                                    name='words after cleaning',
                                    marker=go.bar.Marker(
                                        color='rgb(26, 118, 255)'
                                    )
                                )
                            ],
                            layout=go.Layout(
                                title='AFTER PREPROCESSING',
                                showlegend=True,
                                legend=go.layout.Legend(
                                    x=0,
                                    y=1.0
                                ),
                                margin=go.layout.Margin(l=40, r=0, t=40, b=30)
                            )
                        )

                    ),
                ], className='six columns'),

            ], className='row'),

        ], className="row")

    ], className="twelve columns")


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.probability import FreqDist
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from textblob import TextBlob, Word
from sklearn.pipeline import Pipeline
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re


class CleaningDF:

    def __init__(self, p_df):
        self.p_df = p_df

    def get_info(self):
        return self.p_df.info()

    # cristina. deletes unnecessary features
    def drop_columns(self):
        dropcols = ['id', 'dateadded', 'dateupdated', 'address', 'categories', 'primarycategories', 'keys', 'latitude',
                    'longitude', 'postalcode', 'reviews_date', 'reviews_dateseen', 'reviews_sourceurls',
                    'reviews_usercity', 'reviews_userprovince', 'reviews_username', 'sourceurls', 'websites',
                    'location']

        self.p_df = self.p_df.drop(dropcols, axis=1)
        return self.p_df

    def missing_val(self):
        self.p_df.isnull().values.any()  # cristina
        self.p_df['reviews_text'] = self.p_df['reviews_text'].str.replace('\d+', '')  # cristina. delete numbers
        self.p_df['reviews_text'] = self.p_df['reviews_text'].dropna().reset_index(
            drop=True)  # delete NaN and reindex   #cristina
        self.p_df['reviews_text'] = self.p_df['reviews_text'].astype(str)  # Cristina. To assure all are strings.
        return self.p_df


class PreprocessReview:

    def __init__(self, pr_df):
        self.pr_df = pr_df

    # cristina. Remove Most frequent words
    def common_words(self, wfilter, n_words):
        self.filter = wfilter
        self.n_words = n_words
        all_words = ' '.join([text for text in wfilter])
        all_words = all_words.split()

        # get word frequency
        fdist = FreqDist(all_words)
        words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})  # converts to df

        # selecting top #terms most frequent words and plot
        d = words_df.nlargest(columns="count", n=self.n_words)
        # plt.figure(figsize=(20, 5))
        # ax = sns.barplot(data=d, x="word", y="count")
        # ax.set(ylabel='Count')
        # plt.show()
        return d

    # 1. Kevin. convert to lower case, remove punctuation
    def clean_split_text(self):  # convert to lower case, remove punctuation, tokenize

        self.pr_df['reviews_text_token'] = self.pr_df.apply(lambda row: nltk.word_tokenize(row['reviews_text']),
                                                            axis=1)  # tokenization #cristina
        return self.pr_df

    # Cristina. Remove stop words and reassign to the same column
    def remove_stop_w(self):
        stop_words = stopwords.words('english')  # cristina.
        self.pr_df["reviews_text"] = self.pr_df["reviews_text"].apply(
            lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
        # Cristina. Get ride of numbers
        self.pr_df["reviews_text"] = self.pr_df["reviews_text"].str.replace('\d+', '')
        self.pr_df['reviews_text'] = self.pr_df['reviews_text'].dropna().reset_index(
            drop=True)  # Cristina: delete NaN and reindex

        return self.pr_df

        self.pr_df["reviews_text"] = self.pr_df["reviews_text"].apply(
            lambda x: " ".join(x for x in x.split() if x not in freq))
        self.pr_df['reviews_text'] = self.pr_df['reviews_text'].dropna().reset_index(
            drop=True)  # Cristina: delete NaN and reindex

        return self.pr_df

    # cristina. tokenization - separates words
    def tokenization(self):
        self.pr_df['reviews_text_token'] = self.pr_df.apply(lambda row: nltk.word_tokenize(row['reviews_text']), axis=1)
        return self.pr_df


class Predictors:
    def __init__(self, f_df):
        self.f_df = f_df

    # Cristina. Model and evaluation
    def naivesb(self):

        X_train, X_test, y_train, y_test = train_test_split(self.f_df['reviews_text'],
                                                            self.f_df['reviews_rating'].astype('int'),
                                                            test_size=0.25, random_state=85)
        count_vectorizer = CountVectorizer()  # converts to bags of words and it would remove stop words

        count_train = count_vectorizer.fit_transform(X_train.values)
        count_test = count_vectorizer.transform(X_test.values)

        nb_classifier = MultinomialNB()
        nb_classifier.fit(count_train, y_train)
        pred = nb_classifier.predict(count_test)
        print(metrics.confusion_matrix(y_test, pred, labels=[1, 2, 3, 4, 5]))

        counter = 0
        for review, category in zip(X_test, pred):
            print('%r => %s' % (category, review))
            if (counter == 5):
                break
            counter += 1
        print("Accuracy score Naives Bayes: " + str(metrics.accuracy_score(y_test, pred)))
        return metrics.accuracy_score(y_test, pred)


def main():
    # 1. UPLOAD THE FILE
    df_file = pd.read_csv('datafiniti_hotel_reviews.csv')
    a = df_file.info()

    # 2. CLEANING PROCESS
    clean = CleaningDF(df_file)  # Cristina. instance class CleaningDF()
    df = clean.drop_columns()  # Cristina. Drop features that are not necessary for the analysis
    df = clean.missing_val()  # Cristina - Kevin. Verify and clean missing values and converts to string reviews_text

    # 3. PREPROCESSING
    clean_text = PreprocessReview(df)  # Cristina. instance class PreprocessReview()
    clean_text.common_words(df['reviews_text'], 25)  # Cristina. Shows the frequency of stop words BEFORE removing
    df = clean_text.clean_split_text()  # Cristina - Kevin. Converts to lower case, removes punctuation.

    clean_text.common_words(df['reviews_text'], 25)  # Cristina. Shows the frequency of stop words AFTER removing

    df = clean_text.tokenization()  # Cristina. Tokenization: Convert to strings

    print(df[['reviews_text_lematized']])
    print(df[['reviews_text_token']])

    lsvc = Predictors(df)
    lsvc.linearsvc()

    predictor = Predictors(df)  # Cristina. instance class Predictors()
    prediction_NB = predictor.naivesb()  # Cristina. calls model naives bayes
    print(prediction_NB)


if __name__ == "__main__":  # "Executed when invoked directly"
    main()


