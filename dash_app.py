import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pickle

# Load the saved model using pickle
with open('model/mnb.pkl', 'rb') as f:
    mnb_model = pickle.load(f)

with open('model/svc.pkl', 'rb') as f:
    svc_model = pickle.load(f)

# Load the trained model using pickle
with open('model/rf.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load the trained model using pickle
with open('model/ensemble.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

# Load the transformer using pickle
with open('model/transformer.pkl', 'rb') as f:
    count_vectorizer = pickle.load(f)

# Define the app
app = dash.Dash(__name__)
app.title = "Dota2 Review Classifier"

# Define the style sheet
external_stylesheets = [
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T',
        'crossorigin': 'anonymous'
    }
]

# Add the style sheet to the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout
app.layout = html.Div(className='container mt-5 pt-5', children=[
    html.H1(className='mt-3 mb-5 text-center',
            children='Dota2 Review Classifier'),

    html.Div(className='form-group row mb-5', children=[
        html.Label(className='col-sm-2 col-form-label',
                   children='Enter some text:'),
        html.Div(className='col-sm-10', children=[
            dcc.Input(id='input-box', type='text',
                      className='form-control', value='')
        ])
    ]),

    html.Div(className='form-group row mb-5', children=[
        html.Div(className='col-sm-10 offset-sm-2', children=[
            html.Button('Submit', id='button',
                        className='btn btn-primary', n_clicks=0)
        ])
    ]),

    # html.Div(className='form-group row mb-2', children=[
    #     html.Div(className='col-sm-5', children=[
    #         html.Div(id='output-mnb', className='h5')
    #     ])
    # ]),

    html.Div(className='form-group row mb-2', children=[
        html.Div(className='col-sm-10', children=[
            html.Div(id='output-svc', className='h5')
        ])
    ]),

    html.Div(className='form-group row mb-2', children=[
        html.Div(className='col-sm-10', children=[
            html.Div(id='output-rfc', className='h5')
        ])
    ]),

    html.Div(className='form-group row mb-5', children=[
        html.Div(className='col-sm-10', children=[
            html.Div(id='output-ensemble', className='h5')
        ])
    ]),

    html.Div(className='form-group row mb-2', style={'border': '1px solid black', 'padding': '10px'}, children=[
        html.Div(className='col-sm-10', children=[
            html.Div(id='output-result', className='h2')
        ])
    ]),
])


# Define the callback
@app.callback(
    # [Output('output-mnb', 'children'),
    [Output('output-svc', 'children'),
     Output('output-rfc', 'children'),
     Output('output-ensemble', 'children'),
     Output('output-result', 'children')],
    [Input('button', 'n_clicks')],
    [State('input-box', 'value')])
def update_output(n_clicks, value):
    if n_clicks > 0:
        # Transform the input text using the loaded CountVectorizer object
        userInput = count_vectorizer.transform([value])

        # Use the trained models to make a prediction on the input text
        # nb_pred = mnb_model.predict(userInput)[0]
        # nb_proba = mnb_model.predict_proba(userInput)[0][1]

        svc_pred = svc_model.predict(userInput)[0]
        svc_proba = svc_model.predict_proba(userInput)[0][1]

        rf_pred = rf_model.predict(userInput)[0]
        rf_proba = rf_model.predict_proba(userInput)[0][1]

        ensemble_pred = ensemble_model.predict(userInput)[0]
        ensemble_proba = ensemble_model.predict_proba(userInput)[0][1]

        # Format the prediction as a string
        # if nb_pred == -1:
        #     nb_output_class = 'Negative'
        #     nb_output_class_style = {'color': 'red'}
        #     nb_output_proba = nb_proba
        # else:
        #     nb_output_class = 'Positive'
        #     nb_output_class_style = {'color': 'green'}
        #     nb_output_proba = nb_proba

        if svc_pred == -1:
            svc_output_class = 'Negative'
            svc_output_class_style = {'color': 'red'}
            svc_output_proba = svc_proba
        else:
            svc_output_class = 'Positive'
            svc_output_class_style = {'color': 'green'}
            svc_output_proba = svc_proba

        if rf_pred == -1:
            rf_output_class = 'Negative'
            rf_output_class_style = {'color': 'red'}
            rf_output_proba = rf_proba
        else:
            rf_output_class = 'Positive'
            rf_output_class_style = {'color': 'green'}
            rf_output_proba = rf_proba

        if ensemble_pred == -1:
            ensemble_output_class = 'Negative'
            ensemble_output_class_style = {'color': 'red'}
            ensemble_output_proba = ensemble_proba
        else:
            ensemble_output_class = 'Positive'
            ensemble_output_class_style = {'color': 'green'}
            ensemble_output_proba = ensemble_proba

        # final_pred = svc_pred + rf_pred + ensemble_pred + nb_pred
        final_pred = svc_pred + rf_pred + ensemble_pred

        if final_pred < 0:
            final_output_class = 'Negative'
            final_output_class_style = {'color': 'red'}
        else:
            final_output_class = 'Positive'
            final_output_class_style = {'color': 'green'}

        return (
            # html.Div([
            #     html.Span('Naive Bayes Prediction       : ',
            #               style={'font-weight': 'bold', 'white-space': 'pre'}),
            #     html.Span(nb_output_class, style=nb_output_class_style),
            #     html.Span(f' ({nb_output_proba:.2f})',
            #               style={'font-weight': 'bold'})
            # ]),
            html.Div([
                html.Span('SVC Prediction                     : ',
                          style={'font-weight': 'bold', 'white-space': 'pre'}),
                html.Span(svc_output_class, style=svc_output_class_style),
                html.Span(f' (Probability:{svc_output_proba:.2f})',
                          style={'font-weight': 'bold'})
            ]),
            html.Div([
                html.Span('Random Forest Prediction  : ',
                          style={'font-weight': 'bold', 'white-space': 'pre'}),
                html.Span(rf_output_class, style=rf_output_class_style),
                html.Span(f' (Probability:{rf_output_proba:.2f})',
                          style={'font-weight': 'bold'})
            ]),

            html.Div([
                html.Span('Ensemble (SVC & RFC)        : ',
                          style={'font-weight': 'bold', 'white-space': 'pre'}),
                html.Span(ensemble_output_class,
                          style=ensemble_output_class_style),
                html.Span(f' (Probability:{ensemble_output_proba:.2f})',
                          style={'font-weight': 'bold'})
            ]),

            html.Div([
                html.Span(['Final Desicion for ', html.U(f'\'{value}\''), ' is '],
                          style={'font-weight': 'bold', 'white-space': 'pre'}),
                html.Span(final_output_class,
                          style=final_output_class_style),
                html.Span('!',
                          style={'font-weight': 'bold', 'white-space': 'pre'}),
            ])
        )
    else:
        # return html.Span(''), html.Span(''), html.Span(''), html.Span(''), html.Span('Prediction Result Output')
        return html.Span(''), html.Span(''), html.Span(''), html.Span('Prediction Result Output')


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
