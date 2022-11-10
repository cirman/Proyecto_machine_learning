import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

data = pd.read_csv(".\data.csv",header=0)

def my_density(x,m):
    data = pd.DataFrame({'x': x, 'm': m})
    facet = sns.FacetGrid(data, hue="m",aspect=4)
    facet.map(sns.kdeplot,'x',shade= True)
    facet.set(xlim=(0, data['x'].max()))
    facet.add_legend() 
    plt.show()
#fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div([
    html.Label('Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'Mean Radius', 'value': 'radius_mean'},
            {'label': 'Mean Texture', 'value': 'texture_mean'},
            {'label': 'Mean Perimeter', 'value': 'perimeter_mean'}
        ],
        value='MTL'
    ),
    dcc.Input(value='MTL', type='text'),

    dcc.Graph(
        id='example-graph-2',
        figure=my_density(MTL,"diagnosis")
    )

], style={'columnCount': 2})

if __name__ == '__main__':
    app.run_server(debug=True)