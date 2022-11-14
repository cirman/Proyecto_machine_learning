import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

import plotly.figure_factory as ff



app = dash.Dash(__name__)

data = pd.read_csv(".\data.csv",header=0)
samples=data.iloc[:,2:32] # excluimos la variable de indentificación y la de diagnostico




from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # el reescalador mixmax

scaler.fit(samples)
samples_scaled = scaler.transform(samples)

# importar paquetes
from sklearn.decomposition import PCA

# creamos el modelo y ajustamos
model = PCA()
model.fit(samples_scaled)

# crear un rango que enumere las característica del ACP
caract = range(model.n_components_)


pca = PCA(n_components=4)
principalComponents = pca.fit_transform(samples_scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC 1', 'PC 2','PC 3','PC 4'])


data_new=pd.concat([data[['diagnosis']],principalDf], axis = 1)

data = pd.concat([data,principalDf], axis = 1)



app.layout = html.Div([
    html.Label('Dropdown'),
    dcc.Dropdown(
        id = "continuo",
        options=[
            {'label': 'Mean Radius', 'value': 'radius_mean'},
            {'label': 'Mean Texture', 'value': 'texture_mean'},
            {'label': 'Mean Perimeter', 'value': 'perimeter_mean'},
            {'label' : 'PCA1', 'value' : 'PC 1'},
            {'label' : 'PCA2', 'value' : 'PC 2'},
            {'label' : 'PCA3', 'value' : 'PC 3'},
            {'label' : 'PCA4', 'value' : 'PC 4'}
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