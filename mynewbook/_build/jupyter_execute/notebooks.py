#!/usr/bin/env python
# coding: utf-8

# # EDA (Analisis Exploratorio de Datos)

# In[1]:


# here we will import the libraries used for machine learning

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import numpy as np
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
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
# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then insteda of model_selection we can use cross_validation


# In[2]:


data = pd.read_csv(".\data.csv",header=0)


# Reemplazamos las etiquetas desde strings a numeros: 1 = Maligno; 0 = Beningno

# In[3]:


data.diagnosis = data.diagnosis.replace({"M":1, "B": 0})


# A continuación revisamos la estructura del dataframe y de las variables que contiene

# In[4]:


data.head()


# In[5]:


data.describe()


# Contamos los valores perdidos para cada variable

# In[6]:


data.isnull().sum()


# En este caso el interés es clasificar correctamente los diagnosticos de una base datos. A continuación visualizamos los diagnosticos.

# In[7]:


a = (data
                       .groupby("diagnosis")
                       .agg(frequency=("diagnosis", "count"))
                       .reset_index())

(ggplot(a) +
  geom_bar(aes(x = "diagnosis", y = "frequency"), stat = 'identity'))


# Se puede observar que la mayoría de diagnosticos son de tumores benignos y los malignos se presentan en una menor proporción. A continuación examinamos la distribución de los diagnosticos de acuerdo con otras variables de la base de datos

# In[8]:


plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot=True,fmt='.0%')


# A partir de la matriz de correlacion se observa que hay grupos de variables que se relacionan entre sí. Por ejemplo, las variables radio, perimetro y area tienen una fuerte correlacion entre ellas.

# In[9]:


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

# In[10]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# De acuerdo con los diagramas de densidad, se aprecia que las variables que mejor permiten diferenciar el tipo de tumor, son el perimetro, el area y el radio, ya que en la variable textura, ambos grupos exhiben un alto solapamiento.

# In[11]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5)


# Componentes principales

# A continuación seleccionamos las variables cuantitativas para reducir la dimensionalidad del dataset

# In[12]:


samples=data.iloc[:,2:32] # excluimos la variable de indentificación y la de diagnostico
samples.head(5)


# A continuación escalamos las variables

# In[13]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # el reescalador mixmax

scaler.fit(samples)
samples_scaled = scaler.transform(samples)
print(samples_scaled)


# Ahora examinamos los valores propios para determinar el numero de componentes que debemos extraer

# In[14]:


# importar paquetes
from sklearn.decomposition import PCA

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

# In[15]:


pca = PCA(n_components=4)
principalComponents = pca.fit_transform(samples_scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC 1', 'PC 2','PC 3','PC 4'])
principalDf


# In[16]:


pca.explained_variance_ratio_.sum() # varianza explicada


# Ya obtuvimos las puntuaciones de cada observación en los cuatro componentes. Ahora procedemos a agregar estas puntuaciones en la BD original.

# In[17]:


data_new=pd.concat([data[['diagnosis']],principalDf], axis = 1)
data_new.head()


# Antes de entrenar el modelo nos aseguramos que haya ambos tipos de diganosticos al momento de divir las muestras

# Análisis de clasificación

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(data_new, data_new.diagnosis, random_state=0)
print("Size of training set: {} size of test set: {}".format(X_train.shape[0], X_test.shape[0]))
best_score = 0


# Seleccionamos el modelo con mejor rendimiento

# In[19]:


#for gamma in [0.01, 0.1, 1, 10]:
#    for C in [0.01, 0.1, 1, 10]:
 #       svm = SVC(gamma=gamma, C=C) # Entrena SVC para cada parámetro
  #      scores = cross_val_score(svm, X_train, y_train, cv=5) # Calcula validación cruzada
   #     score = np.mean(scores) # Calcula media de la validación cruzada para precisión
    ##    if score > best_score:
      #      best_score = score
       #     best_parameters = {'C': C, 'gamma': gamma}


# In[20]:


#svm = SVC(**best_parameters)
#svm.fit(X_train, y_train)


# Ya sabemos que el modelo con C= 1 y gamma=0.01 es el que tiene mejor rendimiento. Ahora procedemos a validarlo con el metodo gridSearch

# In[21]:


param_grid = {'C': [0.01, 0.1, 1, 10],
              'gamma': [0.01, 0.1, 1, 10]}
print("Parameter grid:\n{}".format(param_grid))

grid_search = GridSearchCV(SVC(), param_grid, cv=5)

grid_search.fit(X_train, y_train)

print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

