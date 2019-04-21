import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go # Renzo C. For generate  histogram
import base64
import datetime
import io
import dash_table
import pandas as pd

import CleaningProcess




print(dcc.__version__) # 0.6.0 or above is required

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

df = pd.DataFrame([[1,2,3,4]]) #Renzo: global variable for the df. values are manually input to test

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

#Renzo:  Function that Validates the content of the csv file
def parse_contents(contents, filename, date):
    # df2 = pd.DataFrame()
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

     # df2 = df
    return html.Div([

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

    ])

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        CleaningProcess.CleaningDF(list_of_contents)
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children





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

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        CleaningProcess.CleaningDF(list_of_contents)
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



page_3_layout = html.Div([
    html.H1('Text Mining Histogram', style={'color': '#0f2128'}),


    html.Div(id='page-3-content'),
    html.Br(),
    dcc.Link('Go to next page', href='/page-4'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

@app.callback(dash.dependencies.Output('page-3-content', 'children'),
              [dash.dependencies.Input('page-3-radios', 'value')])
def page_3_radios(value):
    return 'You have selected "{}"'.format(value)


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])


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




if __name__ == '__main__':

    app.run_server(debug=True)

