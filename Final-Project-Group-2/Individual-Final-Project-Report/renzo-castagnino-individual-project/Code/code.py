self.pr_df['reviews_text'] = self.pr_df['reviews_text'].dropna().reset_index(drop=True)  # Renzo: delete NaN and reindex

# This code will remove the stop words from the reviews such as “the”, “a”, “an”, “in”)
def remove_stop_w(self):
    stop_words = stopwords.words('english')
    self.pr_df["reviews_text"] = self.pr_df["reviews_text"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    # Get ride of numbers
    self.pr_df["reviews_text"] = self.pr_df["reviews_text"].str.replace('\d+', '')
    self.pr_df['reviews_text'] = self.pr_df['reviews_text'].dropna().reset_index(
        drop=True)  # Renzo: delete NaN and reindex

    return self.pr_df


# renzo.This code will allow to count all the rare words to select how many we will delete. We conclude to delete the 15 most common words ( rather that 10 or 20)
def count_rare_word(self):
    freq = pd.Series(" ".join(self.pr_df['reviews_text']).split()).value_counts()[-15:]
    # freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
    return freq


# renzo . Remove the rare words from the reviews column
def remove_rare_words(self):
    freq = pd.Series(" ".join(self.pr_df['reviews_text']).split()).value_counts()[-15:]
    freq = list(freq.index)
    self.pr_df["reviews_text"] = self.pr_df["reviews_text"].apply(
        lambda x: " ".join(x for x in x.split() if x not in freq))
    self.pr_df['reviews_text'] = self.pr_df['reviews_text'].dropna().reset_index(
        drop=True)  # Renzo: delete NaN and reindex

    return self.pr_df


    #Renzo. converts the word into its root word
    def lematization(self):
        self.pr_df["reviews_text_lematized"] = self.pr_df["reviews_text"]
        self.pr_df["reviews_text_lematized"] = self.pr_df["reviews_text_lematized"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        return self.pr_df

    #Renzo.cutting off the end or the beginning of the word, taking into account a list of common prefixes and suffixes
    def stemming(self):
        st = PorterStemmer()
        self.pr_df["reviews_text_lematized"] = self.pr_df["reviews_text_lematized"].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
        return self.pr_df



    #Renzo. Spelling correction of the reviews
    def spelling_correction(self):
        self.pr_df["reviews_text"] = self.pr_df["reviews_text"].apply(lambda x: str(TextBlob(x).correct()))
        return self.pr_df


    #Renzo
    def linearsvc(self):

        # In this line we are rounding the ratings to improve the accuracy of the model.
        self.f_df['reviews_rating'] = self.f_df['reviews_rating'].round()

        #Creates the training and set set of the reviews to analyse and the ratings to predict
        X_train, X_test, y_train, y_test = train_test_split(self.f_df['reviews_text'], self.f_df['reviews_rating'].astype('int'), test_size = 0.25, random_state=53)

        #The model consisst of 3 steps:
        #1. TFID vectorizer takes the feature and creates a matrix of the bag of words. This algorithm gives every words a certaing weight. Words that are very frequenty will get a lower  rating
        # The N-gram will tell us how many words to consider, for example 1-2 look separate words but also pair of words.

        #2.Chi-squared will take out of all the bag of words, we are selecting the best features. In this case the best 8,000 features. This stochastic algortihhm gives weighting so the features are very dependent from each other.
        #3. The las part is the classifier LinearSVC, with lasso penalty.
        pipeline = Pipeline([
            ('vect', TfidfVectorizer(ngram_range=(1,3), stop_words=None, sublinear_tf=True)),
            ('chi', SelectKBest(chi2, k=8000)),
            ('clf', LinearSVC(C=1.0, penalty='l1',max_iter=3000, dual=False))
                            ])

        #With this we have our model ready. The pipeline will pass each of the results thru the pipeline created above.
        model = pipeline.fit(X_train, y_train)

        #In this part we get the instances of the pipeline process
        vectorizer = model.named_steps['vect']
        chi = model.named_steps['chi']
        clf = model.named_steps['clf']
        feature_names = vectorizer.get_feature_names()
        feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
        feature_names = np.asarray(feature_names)

        # We create the target for the rating
        target_names = ['1', '2', '3', '4', '5']

        print("The top keywords are: ")
        # This will print the best keywords for each class we have
        for i, label in enumerate(target_names):
            top10 = np.argsort(clf.coef_[i])[-10:]
            print("%s: %s" %(label, " ".join(feature_names[top10])))

        print("Accuracy score: " + str(model.score(X_test, y_test)))
        print(model.predict(['that was an awesome place. great food!']))

        return model.score(X_test,y_test)



    df = clean_text.remove_stop_w()                       # Renzo. It removes stop words from reviews_text

    cwc = clean_text.count_rare_word()                    # Renzo. Count the rare words in the reviews. I tried with :10, then with :-20
    df = clean_text.remove_rare_words()                   # Renzo. This will clean the rare words from the reviews column
    #df = clean_text.spelling_correction()                # Renzo. This will do the spelling correction. SLOW PROCESS
    df = clean_text.tokenization()                        # Renzo. Tokenization: Convert to strings
    df = clean_text.lematization()                        # Renzo. Converts the word into its root word
    df = clean_text.stemming()                            # Renzo. This will do the stemming process




#Renzo C: This the main page. It will show the links to the other pages
index_page = html.Div([
    html.H2('Hotel Reviews - Text Mining', style={'textAlign': 'left', 'color': '#0f2128'}),
    dcc.Link('Upload Data', href='/page-1'),
    html.Br(),
    dcc.Link('Preprocessing', href='/page-2'),
    html.Br(),
    dcc.Link('Histogram', href='/page-3'),
    html.Br(),
    dcc.Link('Word Map', href='/page-4'),
])



#Renzo: This is the layout of the 1st page. Upload the csv file
page_1_layout = html.Div([
    html.H2('Hotel Reviews - Upload your file', style={'textAlign': 'left', 'color': '#0f2128'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '30%',
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

    html.Br(),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    dcc.Link('Preprocessing', href='/page-2'),
    html.Br(),

    html.Div(id='output-data-upload'),
])

#Renzo: This is the layout of the 2nd page. Upload the csv file
page_2_layout = html.Div([
    html.H2('Preprocessing - pending to assign uploaded csv to global variable', style={'textAlign': 'left', 'color': '#0f2128'}),

    dash_table.DataTable(
            data=df.to_dict('rows'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={
                'maxHeight': '600px',
                'maxWidth': '1800px',
                'overflowY': 'scroll',
                #'overflowX': 'scroll'
            },
        ),
    html.Hr(),  # horizontal line

    html.Br(),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    dcc.Link('Go to Statistics', href='/page-3'),
    html.Br(),

    html.Div(id='show_data'),
])

@app.callback(Output('show_data', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])




# Renzo: Redirect to each of the pages
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    else:
        return index_page


