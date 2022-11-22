#!/usr/bin/env python
# coding: utf-8

# # Clasificación de tumores benignos y malignos

# ## EDA (Analisis Exploratorio de Datos)

# Para este ejercicio de aprendizaje sobre tecnicas de aprendizaje supervizado se utilizarán los siguientes datos: https://www.kaggle.com/code/gargmanish/basic-machine-learning-with-cancer/data
# Este conjunto de datos contiene información sobre caracteristicas las de tumores detectados en pacientes que fueron diagnosticados como malignos o benignos. A continuación se importan las librerias que se usarán en la actividad. 

# In[1]:


# here we will import the libraries used for machine learning

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import numpy as np
import random
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model
from plotnine import * # incluye funciones de ggplot
from sklearn.svm import SVC # support vector machine
from sklearn.model_selection import cross_val_score # validacion cruzada
from sklearn.model_selection import GridSearchCV # grid
from sklearn.pipeline import make_pipeline
import mglearn
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix


# A contunuación importamos los datos usando la función read_csv de la librería pandas

# In[2]:


data = pd.read_csv(".\data.csv",header=0)


# Reemplazamos las etiquetas desde strings a numeros: 1 = Maligno; 0 = Beningno

# In[3]:


data.diagnosis = data.diagnosis.replace({"M":1, "B": 0})


# A continuación revisamos la estructura del dataframe y de las variables que contiene

# In[4]:


data.head()


# Contamos los valores perdidos para cada variable

# In[5]:


pd.DataFrame(data.isnull().sum()).transpose()


# Se puede observar que las variables no presentan valores perdidos a excepción de la ultima columna, la cual será excluida de los análisis

# Considerando que el interés en este ejercicio es clasificar correctamente los diagnosticos de una base datos, a continuación visualizamos los diagnosticos.

# In[6]:


a = (data
                       .groupby("diagnosis")
                       .agg(frequency=("diagnosis", "count"))
                       .reset_index())

(ggplot(a) +
  geom_bar(aes(x = "diagnosis", y = "frequency"), stat = 'identity'))


# Se puede observar que la mayoría de diagnosticos son de tumores benignos y los malignos se presentan en una menor frecuencia. A continuación examinamos la distribución de los diagnosticos de acuerdo con otras variables de la base de datos.

# Considerando que se tiene un número elevado de variables que registran información sobre los tumores, procedemos a examinar si se pueden agrupar teniendo en cuenta sus correlaciones.

# In[7]:


plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot=True,fmt='.0%')


# A partir de la matriz de correlacion se observa que hay grupos de variables que se relacionan entre sí. Por ejemplo, las variables radio, perimetro y area tienen una fuerte correlacion entre ellas.

# In[8]:


facet = sns.FacetGrid(data, hue="diagnosis",aspect=4)
facet.map(sns.kdeplot,'radius_mean',shade= True)
facet.set(xlim=(0, data['radius_mean'].max()))
facet.add_legend() 
plt.show()

facet = sns.FacetGrid(data, hue="diagnosis",aspect=4)
facet.map(sns.kdeplot,'texture_mean',shade= True)
facet.set(xlim=(0, data['texture_mean'].max()))
facet.add_legend() 
plt.xlim(10,40)


# Los tumores malignos, presentan un radio promedio mayor en comparación con los tumores malignos. Mientras que con respecto a la textura (desviación estándar de los valores de la escala de grises), los tumores malignos también muestran una puntuación promedio más alta, que los tumores benignos.

# In[9]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# De acuerdo con los diagramas de densidad, se aprecia que las variables que mejor permiten diferenciar el tipo de tumor, son el perimetro, el area y el radio, ya que en la variable textura, ambos grupos exhiben un alto solapamiento.

# In[10]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5)


# ## Componentes principales

# A continuación seleccionamos las variables cuantitativas para reducir la dimensionalidad del dataset

# In[11]:


samples=data.iloc[:,2:32] # excluimos la variable de indentificación y la de diagnostico


# A continuación escalamos las variables

# In[12]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # el reescalador mixmax

scaler.fit(samples)
samples_scaled = pd.DataFrame(scaler.transform(samples),columns=samples.columns)
samples_scaled


# Ahora examinamos los valores propios para determinar el numero de componentes que debemos extraer

# In[13]:


# creamos el modelo y ajustamos
model = PCA()
model.fit(samples_scaled)

# crear un rango que enumere las característica del ACP
caract = range(model.n_components_)

# grafiquemos la varianza explicada del modelo ACP
plt.bar(caract,model.explained_variance_)
plt.xticks(caract)
plt.ylabel('Varianza')
plt.xlabel('variables del ACP')
plt.show()


# Analizando la varianza explicada por cada componente, parece suficiente extraer cuatro componentes

# In[14]:


pca = PCA(n_components=4)
principalComponents = pca.fit_transform(samples_scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC 1', 'PC 2','PC 3','PC 4'])
principalDf


# In[15]:


pca.explained_variance_ratio_.sum() # varianza explicada


# La varianza total explicada por las cuatro componentes es 83.90% lo cual significa que hubo poca perdida de información si se tiene en cuenta que se redujeron más de 20 dimensiones de la base de datos.

# In[16]:


x = pd.DataFrame(
    data=np.transpose(pca.components_),
    columns=["PC1","PC2","PC3","PC4"],
    index=samples.columns
)
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.heatmap(x, annot=True, cmap='Spectral', linewidths=.5)
plt.show()


# Examinando la matriz de cargas se puede observar que en el componente 1 las variables con mayor aporte de información son concave points_worst y concave points_means. En la segunda componente las variables con mayor aporte son fractal dimension_mean, fractal dimension_mean y radius_mean. La tercera componente es altamente representada por la información de las variables texture_worst, texture_mean y texture_se. Por su parte en la cuarta componente las variables con mayores cargas son texture_se, symmetry_se y smoothness_worst.

# ## Análisis de clasificación

# Un  aspecto importante de los análisis de clasificación es contar con un conjunto de datos que nos permitan evaluar el score del modelo. Para ello se dividirán los datos en entrenamiento y testeo usando la función train_test_split. En este caso se tuvo en cuenta el diagnostico para realizar el split de manera estratificada y así garantizar que las dos muestras tengan observaciones con ambos tipos de diagnostico.

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,2:-1], data.diagnosis, random_state=0)
print("Size of training set: {} size of validation set: {}".format(X_train.shape[0], X_test.shape[0]))


# ### Maquina de soporte vectorial (SVM)

# A continuación usamos maquina de soporte vectorial (SVM) como metodo de clasificación. Usaremos inicialmente los valores 1, 10, 100, 1000, 10000 y 100000 para Gamma y 0.00001, 0.0001, 0.001, 0.01, 0.1 y 1 para C para evaluar diferentes modelos SVM y elegir el que tenga mejor score. Para optimizar los procesos de computo se realizarán los análisis de clasificación usando pipelines.

# In[18]:


param_grid = {'svm__C': [1, 10, 100, 1000, 10000, 100000], 
              'svm__gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}

pipe = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=4)), ("svm", SVC())])
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: {}".format(grid.best_params_))


# Se puede observar que el modelo es capaz de clasificar correctamente 96% de los datos de testeo. Siendo SVM más eficiente cuando gamma = 0.01 y C = 100.

# In[19]:


results = pd.DataFrame(grid.cv_results_)
scores = np.array(results.mean_test_score).reshape(6, 6)
mglearn.tools.heatmap(scores, 
                      xlabel='gamma', 
                      xticklabels=param_grid['svm__gamma'], 
                      ylabel='C', 
                      yticklabels=param_grid['svm__C'], 
                      cmap="viridis");


# En el mapa de calor se observan los scores que se obtiene al usar SVM variando cada uno de los parametros. Considerando que las zonas iluminadas se encuentran en el centro del mapa no parece necesario ajustar los valores de gamma y C.

# In[20]:


pred=grid.predict(X_test)
scores_image = mglearn.tools.heatmap(confusion_matrix(y_test, pred), 
                                     xlabel='Predicted label',
                                     ylabel='True label', 
                                     xticklabels=[0,1],
                                     yticklabels=[0,1],
                                     cmap=plt.cm.gray_r, 
                                     fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()


# Una vez seleccionado el mejor modelo, se evalúan examinan los casos clasificados correctamente. Se observa que de los 53 casos con cancer detectó 49 y los 90 casos sin cancer detectó 88.

# In[21]:


fpr, tpr, thresholds = roc_curve(y_test, grid.decision_function(X_test))
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=500, height=500)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# Una vez se observan los casos clasificados correctamente, es necesario examinar la curva ROC la cual muestra la relación entre las clasificaciones correctas e incorrectas. Asimismo, el AUC es un valor que oscila entre 0 y 1. Valores cercanos a 1 indican mayor calidad del modelo. En este caso el AUC de 0.996 muestra que el modelo resulta bastante bueno.

# ### Modelo de regresión logística

# Ya se evaluó la capacidad de clasificar de SVM siendo esta de 96% para los datos de testeo. A continuación se realiza el mismo procedimiento, pero implementendo modelos de regresión logística para determinar si este clasifica mejor que SVM.

# In[22]:


param_grid={'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

pipe = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=4)), ("logisticregression", LogisticRegression())])
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs = -1)
grid.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: {}".format(grid.best_params_))


# Los resultados muestran que este modelo clasifica correctamente 97% de los datos de testeo, 1% más que el modelo SVM.

# In[23]:


pred=grid.predict(X_test)
scores_image = mglearn.tools.heatmap(confusion_matrix(y_test, pred), 
                                     xlabel='Predicted label',
                                     ylabel='True label', 
                                     xticklabels=[0,1],
                                     yticklabels=[0,1],
                                     cmap=plt.cm.gray_r, 
                                     fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()


# La matriz de confusión muestra que el modelo logístico solo tuvo 3 falsos negativos y 2 falsos positivos.

# In[24]:


fpr, tpr, thresholds = roc_curve(y_test, grid.decision_function(X_test))
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=500, height=500)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# El AUC del modelo logístico muestra ser superior al de SVM.

# In[25]:


random.seed(10)
param_grid = {'Gradient_boosting__n_estimators': [10, 20, 30, 40, 50, 55, 60, 65, 80]}

pipe = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=4)), ("Gradient_boosting", xgb.XGBClassifier(objective="binary:logistic", booster='gblinear', learning_rate =0.1, eval_metric="auc"))])

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs = -1)
grid.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: {}".format(grid.best_params_))


# Al igual que el modelo logístico, este modelo clasifica correctamente el 97% de los datos de testeo.

# In[26]:


fpr, tpr, thresholds = roc_curve(y_test, grid.predict_proba(X_test)[:,1])
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=500, height=500)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# In[27]:


pred=grid.predict(X_test)
scores_image = mglearn.tools.heatmap(confusion_matrix(y_test, pred), 
                                     xlabel='Predicted label',
                                     ylabel='True label', 
                                     xticklabels=[0,1],
                                     yticklabels=[0,1],
                                     cmap=plt.cm.gray_r, 
                                     fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()


# La matriz de confusión muestra que el modelo XGB solo tuvo 4 falsos negativos (una más que el modelo logístico) y 2 falsos positivos.

# In[28]:


pipe = Pipeline([('preprocessing', MinMaxScaler()), ("PCA", PCA(n_components=4)), ('classifier', RandomForestClassifier())])

param_grid = [{'classifier': [RandomForestClassifier(n_estimators=100)],

               'preprocessing': [None], 'classifier__max_features': [1, 2, 3, 4]}]

grid = GridSearchCV(pipe, param_grid, cv=5)

grid.fit(X_train, y_train)

print("Best params:\n{}\n".format(grid.best_params_))

print("Best cross-validation score: {:.2f}".format(grid.best_score_))

print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))


# Al ensayar con el Random Forest model, este clasifica correctamente el 94% de los datos de testeo (porcentaje menor que los modelos previos).

# In[29]:


fpr, tpr, thresholds = roc_curve(y_test, grid.predict_proba(X_test)[:,1])
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=500, height=500)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# In[30]:


pred=grid.predict(X_test)
scores_image = mglearn.tools.heatmap(confusion_matrix(y_test, pred), 
                                     xlabel='Predicted label',
                                     ylabel='True label', 
                                     xticklabels=[0,1],
                                     yticklabels=[0,1],
                                     cmap=plt.cm.gray_r, 
                                     fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()


# La matriz de confusión del Random Forest model muestra 4 falsos negativos y 4 falsos positivos (mas que el XGB, M. Logistico y SVM).
