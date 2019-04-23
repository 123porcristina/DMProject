import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import CleaningProcess as dmproject

import dash_core_components as dcc
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df_file = pd.read_csv('datafiniti_hotel_reviews.csv')

# 2. CLEANING PROCESS
clean = dmproject.CleaningDF(df_file)
df = clean.drop_columns()                             # Cristina. Drop features that are not necessary for the analysis
df = clean.missing_val()                              # Cristina - Kevin. Verify and clean missing values and converts to string reviews_text


# 3. PREPROCESSING
clean_text = dmproject.PreprocessReview(df_file)
freq=clean_text.common_words(df['reviews_text'],25)        # Cristina. Shows the frequency of stop words BEFORE removing
df = clean_text.clean_split_text()                    # Cristina - Kevin. Converts to lower case, removes punctuation.
df = clean_text.remove_stop_w()                       # Renzo. It removes stop words from reviews_text
cwc = clean_text.count_rare_word()                    # Renzo. Count the rare words in the reviews. I tried with :10, then with :-20
df = clean_text.remove_rare_words()                   # Renzo. This will clean the rare words from the reviews column
freqA=clean_text.common_words(df['reviews_text'], 25)       # Cristina. Shows the frequency of stop words AFTER removing



app.layout = html.Div([
    html.Div([

        html.Div(children=[
            html.H2(children='Text Analysis Hotel Reviews in the U.S',
                    style={'textAlign': 'center', 'color': '#0f2128'},
                    className= "twelve columns"), #title occupies 9 cols

            html.Div(children=''' 
                        Dash: Text Analysis Hotel Reviews in the U.S.
                        ''',
                     className="nine columns")#this subtitle occupies 9 columns
        ], className = "row"),

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

        html.Div([
            html.Div([
                dcc.Graph(
                    # style={'height': 300},
                    id='example-graph'
                )
            ], className="six columns"),  # six columns for 1st graphic
            html.Div([
                dcc.Graph(
                    id='example-graph2'
                )
            ], className="six columns")  # 6 columns for this 2nd graphic in total can be max 12 cols
        ], className= "row"),

        html.Div([
            html.Div([
                dcc.Graph(
                    id='example-graph3'
                )
            ])
        ], className="row")
    ])
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        #html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('rows'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'scroll', 'maxHeight': '400px','overflowY': 'scroll'},
            style_cell={'padding': '5px',
                        'whiteSpace': 'no-wrap',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'maxWidth': 0,
                        'height': 30,
                        'textAlign': 'left'},
                       style_header = {'backgroundColor': 'white',
                                       'fontWeight': 'bold',
                                       'color': 'black'}
        ),

    ], className="nine columns", style = {'margin-top': '35',
                                           'margin-left': '15',
                                           'border': '1px solid #C6CCD5'})


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

@app.callback(Output('example-graph', 'figure'),
             [Input('upload-data', 'children')])
def get_before_frequency(DataTable):

    #convert colums to list
    words = freq['word'].tolist()
    twords= freq['count'].tolist()

    figure=go.Figure(
        data = [
            go.Bar(
                x=words,
                y=twords,
                name='Before Preprocessing',
                marker=go.bar.Marker(color='rgb(55, 83, 109)'
                )
            ),
        ],
        layout = go.Layout(
            title='Before Cleaning',
            showlegend=True,
            legend=go.layout.Legend(
                x=0,
                y=1.0
            ),
            margin=go.layout.Margin(l=40, r=0, t=40, b=30)
        )
    )
    return figure


@app.callback(Output('example-graph2', 'figure'),
             [Input('upload-data', 'children')])
def get_after_frequency(DataTable):

    wordsA = freqA['word'].tolist()
    twordsA = freqA['count'].tolist()

    figure = go.Figure(
        data=[

            go.Bar(
                x=wordsA,
                y=twordsA,
                name='words after cleaning',
                marker=go.bar.Marker(
                    color='rgb(26, 118, 255)'
                )
            )
        ],
        layout=go.Layout(
            title='After Cleaning',
            showlegend=True,
            legend=go.layout.Legend(
                x=0,
                y=1.0
            ),
            margin=go.layout.Margin(l=40, r=0, t=40, b=30)
        )
    )

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)