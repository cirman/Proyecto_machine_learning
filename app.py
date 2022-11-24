import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
import mglearn
from sklearn.svm import SVC # support vector machine
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix


import plotly.figure_factory as ff



app = dash.Dash(__name__)

data = pd.read_csv(".\data.csv",header=0)
samples=data.iloc[:,2:32] # excluimos la variable de indentificación y la de diagnostico
#datos = data
data.diagnosis = data.diagnosis.replace({"M":1, "B": 0})
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,2:-1], data.diagnosis, random_state=0)



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

x = pd.DataFrame(
    data=np.transpose(pca.components_),
    columns=["PC1","PC2","PC3","PC4"],
    index=samples.columns
)

calor = px.imshow(x, text_auto=True, aspect="auto")




data_new=pd.concat([data[['diagnosis']],principalDf], axis = 1)

data = pd.concat([data,principalDf], axis = 1)



app.layout =  dcc.Tabs([
    dcc.Tab(label='EDA', children=[
        html.Div([
            dbc.Row([
                    dbc.Col(html.H1("Distribucion de densidad de las variables"), className="mb-2")
                ]),
            dbc.Row([
                dbc.Col(html.H6(children='Aqui podemos ver la distribucion de densidad para cada una de las variables tomadas en cuenta en el proyecto, como tambien para cada una de las 4 variables de PCA obtenidas luego de hacer el Analisis de componentes principales'), className="mb-4")
            ]),
            html.Label('Escoja una variable'),
    
    
            html.Div([
                dcc.Dropdown(
                    id = "continuo",
                    options=[
                        {'label': 'Mean Radius', 'value': 'radius_mean'},
                        {'label': 'Mean Texture', 'value': 'texture_mean'},
                        {'label': 'Mean Perimeter', 'value': 'perimeter_mean'},
                        {'label': 'Mean Area', 'value': 'area_mean'},
                        {'label': 'Mean Smoothness', 'value': 'smoothness_mean'},
                        {'label': 'Mean Compactness', 'value': 'compactness_mean'},
                        {'label': 'Mean Concavity', 'value': 'concavity_mean'},
                        {'label': 'Mean Concave Points', 'value': 'concave points_mean'},
                        {'label': 'Mean Symmetry', 'value': 'symmetry_mean'},
                        {'label': 'Mean Fractal Dimension', 'value': 'fractal_dimension_mean'},
                        {'label': 'Standard Error Radius', 'value': 'radius_se'},
                        {'label': 'Standard Error Texture', 'value': 'texture_se'},
                        {'label': 'Standard Error Perimeter', 'value': 'perimeter_se'},
                        {'label': 'Standard Error Area', 'value': 'area_se'},
                        {'label': 'Standard Error Smoothness', 'value': 'smoothness_se'},
                        {'label': 'Standard Error Compactness', 'value': 'compactness_se'},
                        {'label': 'Standard Error Concavity', 'value': 'concavity_se'},
                        {'label': 'Standard Error Concave Points', 'value': 'concave points_se'},
                        {'label': 'Standard Error Symmetry', 'value': 'symmetry_se'},
                        {'label': 'Standard Error Fractal Dimension', 'value': 'fractal_dimension_se'},
                        {'label': 'Worst Radius', 'value': 'radius_worst'},
                        {'label': 'Worst Texture', 'value': 'texture_worst'},
                        {'label': 'Worst Perimeter', 'value': 'perimeter_worst'},
                        {'label': 'Worst Area', 'value': 'area_worst'},
                        {'label': 'Worst Smoothness', 'value': 'smoothness_worst'},
                        {'label': 'Worst Compactness', 'value': 'compactness_worst'},
                        {'label': 'Worst Concavity', 'value': 'concavity_worst'},
                        {'label': 'Worst Concave Points', 'value': 'concave points_worst'},
                        {'label': 'Worst Symmetry', 'value': 'symmetry_worst'},
                        {'label': 'Worst Fractal Dimension', 'value': 'fractal_dimension_worst'},
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
            ]),


            html.Div([
                html.Div(
                    [html.H6("Mapa de calor de las variables respecto al analisis de componentes principales", className="graph__title")]
                ),
                dcc.Graph(
                    id='graph_calor',
                    figure = calor
                )
            ])
        ])
    ]),
    dcc.Tab(label='Modelamiento', children=[
        html.Div([
            dbc.Row([
                dbc.Col(html.H1("Modelos de Prediccion del diagnostico"), className="mb-2")
            ]),
            dbc.Row([
                dbc.Col(html.H6(children='Aqui podemos comparar los desempenos de diferentes modelos para predecir si un tumor de cancer de seno es benigno o maligno'), className="mb-4")
            ]),

            html.Label('Escoja un modelo para ver su rendimiento'),

            dcc.Dropdown(
                id = "model_select",
                options=[
                    {'label': 'Support Vector Machine', 'value': 'svm'},
                    {'label': 'Regresion Logistica', 'value': 'logisticregression'},
                    {'label': 'XGBOOST', 'value': 'Gradient_boosting'},
                    {'label': 'Random Forest', 'value' : 'RandomForest'}
                ],
                value='svm'
            ),

            dbc.Row([
                dbc.Col(html.H6(children='Una curva ROC es una representacion grafica de la sensibilidad frente a la especificidad  para un sistema de clasificacion binario segun varia su umbral de discriminacion'), className="mb-4")
            ]),

            dcc.Graph(
                id='roc_curve'
            ),
            
            dbc.Row([
                dbc.Col(html.H6(children='Una tabla de contingencia es una tabla que cuenta las observaciones por múltiples variables categóricas. En general, el interés se centra en estudiar si existe alguna asociación entre filas y columnas, y se calcula la intensidad de dicha asociación.'), className="mb-4")
            ]),

            html.Label('Aqui podemos ver su matriz de contingencia'),

            dcc.Graph(
                id='contingency'
            ),

        ])
    ])
])

@app.callback(
    Output('example-graph-2', 'figure'),
    Input('continuo', 'value'))
def update_graph(continuo):
    datos = data[[continuo,'diagnosis']]
    hist_data = [datos[datos['diagnosis']==1][continuo].values, datos[datos['diagnosis']==0][continuo].values]
    group_labels = ['Maligno', 'Beningno']

    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=['Red','Blue'])
    fig.update_layout(title_text='Curva de densidad de '+continuo)

    return fig

@app.callback(
    Output('roc_curve', 'figure'),
    Input('model_select', 'value'))
def update_roc(model_select):
    if(model_select=="RandomForest"):
        pipe = Pipeline([('preprocessing', MinMaxScaler()), ("PCA", PCA(n_components=4)), ('classifier', RandomForestClassifier())])
        param_grid = [{'classifier': [RandomForestClassifier(n_estimators=100)],
        'preprocessing': [None], 'classifier__max_features': [1, 2, 3, 4]}]
        grid = GridSearchCV(pipe, param_grid, cv=5)
        grid.fit(X_train, y_train)
        fpr, tpr, thresholds = roc_curve(y_test, grid.predict_proba(X_test)[:,1])
        roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=500, height=500)
        roc.update_yaxes(scaleanchor="x", scaleratio=1)
        roc.update_xaxes(constrain='domain')
        return roc

    if(model_select=="logisticregression"):
        param_grid={'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
        pipe = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=4)), ("logisticregression", LogisticRegression())])
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs = -1)
        grid.fit(X_train, y_train)
        fpr, tpr, thresholds = roc_curve(y_test, grid.decision_function(X_test))
        roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=500, height=500)
        roc.update_yaxes(scaleanchor="x", scaleratio=1)
        roc.update_xaxes(constrain='domain')
        return roc
    
    if(model_select=="svm"):
        param_grid = {'svm__C': [1, 10, 100, 1000, 10000, 100000], 
              'svm__gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
        pipe = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=4)), ("svm", SVC())])
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
        grid.fit(X_train, y_train)
        fpr, tpr, thresholds = roc_curve(y_test, grid.decision_function(X_test))
        roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=500, height=500)
        roc.update_yaxes(scaleanchor="x", scaleratio=1)
        roc.update_xaxes(constrain='domain')
        return roc
    
    if(model_select=="Gradient_boosting"):
        param_grid = {'Gradient_boosting__n_estimators': [10, 20, 30, 40, 50, 55, 60, 65, 80]}
        pipe = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=4)), ("Gradient_boosting", xgb.XGBClassifier(objective="binary:logistic", booster='gblinear', learning_rate =0.1, eval_metric="auc"))])
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs = -1)
        grid.fit(X_train, y_train)
        fpr, tpr, thresholds = roc_curve(y_test, grid.predict_proba(X_test)[:,1])
        roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=500, height=500)
        roc.update_yaxes(scaleanchor="x", scaleratio=1)
        roc.update_xaxes(constrain='domain')
        return roc

@app.callback(
    Output('contingency', 'figure'),
    Input('model_select', 'value'))
def update_cont(model_select):
    if(model_select=="svm"):
        param_grid = {'svm__C': [1, 10, 100, 1000, 10000, 100000], 
              'svm__gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
        pipe = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=4)), ("svm", SVC())])
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
        grid.fit(X_train, y_train)
        pred=grid.predict(X_test)
        cont = px.imshow(confusion_matrix(y_test, pred), text_auto=True, aspect="auto")
        return cont

    if(model_select=="logisticregression"):
        param_grid={'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
        pipe = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=4)), ("logisticregression", LogisticRegression())])
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs = -1)
        grid.fit(X_train, y_train)
        pred=grid.predict(X_test)
        cont = px.imshow(confusion_matrix(y_test, pred), text_auto=True, aspect="auto")
        return cont

    if(model_select=="RandomForest"):
        pipe = Pipeline([('preprocessing', MinMaxScaler()), ("PCA", PCA(n_components=4)), ('classifier', RandomForestClassifier())])
        param_grid = [{'classifier': [RandomForestClassifier(n_estimators=100)],
        'preprocessing': [None], 'classifier__max_features': [1, 2, 3, 4]}]
        grid = GridSearchCV(pipe, param_grid, cv=5)
        grid.fit(X_train, y_train)
        pred=grid.predict(X_test)
        cont = px.imshow(confusion_matrix(y_test, pred), text_auto=True, aspect="auto")
        return cont

    if(model_select=="Gradient_boosting"):
        param_grid = {'Gradient_boosting__n_estimators': [10, 20, 30, 40, 50, 55, 60, 65, 80]}
        pipe = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=4)), ("Gradient_boosting", xgb.XGBClassifier(objective="binary:logistic", booster='gblinear', learning_rate =0.1, eval_metric="auc"))])
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs = -1)
        grid.fit(X_train, y_train)
        pred=grid.predict(X_test)
        cont = px.imshow(confusion_matrix(y_test, pred), text_auto=True, aspect="auto")
        return cont
    
    




if __name__ == '__main__':
    app.run_server(debug=True)