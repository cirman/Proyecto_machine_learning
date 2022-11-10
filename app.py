import matplotlib
matplotlib.use('TkAgg')
import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import seaborn as sns

import matplotlib.pyplot as plt

app = dash.Dash(__name__)

data = pd.read_csv(".\data.csv",header=0)


#fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

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
    facet = sns.FacetGrid(datos, hue='diagnosis',aspect=4)
    facet.map(sns.kdeplot,continuo,shade= True)
    facet.set(xlim=(0, datos[continuo].max()))
    facet.add_legend()
    plt.show()


if __name__ == '__main__':
    app.run_server(debug=True)