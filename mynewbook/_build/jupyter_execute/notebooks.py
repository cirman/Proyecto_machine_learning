#!/usr/bin/env python
# coding: utf-8

# # Content with notebooks
# 
# You can also create content with Jupyter Notebooks. This means that you can include
# code blocks and their outputs in your book.
# 
# ## Markdown + notebooks
# 
# As it is markdown, you can embed images, HTML, etc into your posts!
# 
# ![](https://myst-parser.readthedocs.io/en/latest/_static/logo-wide.svg)
# 
# You can also $add_{math}$ and
# 
# $$
# math^{blocks}
# $$
# 
# or
# 
# $$
# \begin{aligned}
# \mbox{mean} la_{tex} \\ \\
# math blocks
# \end{aligned}
# $$
# 
# But make sure you \$Escape \$your \$dollar signs \$you want to keep!
# 
# ## MyST markdown
# 
# MyST markdown works in Jupyter Notebooks as well. For more information about MyST markdown, check
# out [the MyST guide in Jupyter Book](https://jupyterbook.org/content/myst.html),
# or see [the MyST markdown documentation](https://myst-parser.readthedocs.io/en/latest/).
# 
# ## Code blocks and outputs
# 
# Jupyter Book will also embed your code blocks and output in your book.
# For example, here's some sample Matplotlib code:

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
# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then insteda of model_selection we can use cross_validation


# In[2]:


data = pd.read_csv(".\data.csv",header=0)


# A continuación revisamos la estructura del dataframe y de las variables que contiene

# In[3]:


data.head()


# In[4]:


data.describe()


# En este caso el interés es clasificar correctamente los diagnosticos de una base datos. A continuación visualizamos los diagnosticos.

# In[5]:


a = (data
                       .groupby("diagnosis")
                       .agg(frequency=("diagnosis", "count"))
                       .reset_index())

(ggplot(a) +
  geom_bar(aes(x = "diagnosis", y = "frequency"), stat = 'identity'))


# Se puede observar que la mayoría de diagnosticos son de tumores benignos y los malignos se presentan en una menor proporción. A continuación examinamos la distribución de los diagnosticos de acuerdo con otras variables de la base de datos

# In[6]:


plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot=True,fmt='.0%')


# A partir de la matriz de correlacion se observa que hay grupos de variables que se relacionan entre sí. Por ejemplo, las variables radio, perimetro y area tienen una fuerte correlacion entre ellas.

# In[5]:


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

# In[3]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# De acuerdo con los diagramas de densidad, se aprecia que las variables que mejor permiten diferenciar el tipo de tumor, son el perimetro, el area y el radio, ya que en la variable textura, ambos grupos exhiben un alto solapamiento.

# In[4]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5)

