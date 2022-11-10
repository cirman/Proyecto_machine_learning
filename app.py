import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

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

], style={'columnCount': 2})

@app.callback(
    Output('example-graph-2', 'figure'),
    Input('continuo', 'value'))
def update_graph(continuo):
    fig = my_density(continuo,'diagnosis')
    return fig
if __name__ == '__main__':
    app.run_server(debug=True)