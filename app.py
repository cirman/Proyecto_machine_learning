import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

import plotly.figure_factory as ff



app = dash.Dash(__name__)

data = pd.read_csv(".\data.csv",header=0)



app.layout = html.Div([
    html.Label('Dropdown'),
    dcc.Dropdown(
        id = "continuo",
        options=[
            {'label': 'Mean Radius', 'value': 'radius_mean'},
            {'label': 'Mean Texture', 'value': 'texture_mean'},
            {'label': 'Mean Perimeter', 'value': 'perimeter_mean'}
        ],
        value='radius_mean'
    ),
    dcc.Graph(
        id='example-graph-2'
    )

])

@app.callback(
    Output('example-graph-2', 'figure'),
    Input('continuo', 'value'))
def update_graph(continuo):
    datos = data[[continuo,'diagnosis']]
    hist_data = [datos[datos['diagnosis']=="M"][continuo].values, datos[datos['diagnosis']=="B"][continuo].values]
    group_labels = ['Maligno', 'Beningno']

    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=['Red','Blue'])
    fig.update_layout(title_text='Curva de densidad de '+continuo)

    return fig



if __name__ == '__main__':
    app.run_server(debug=True)