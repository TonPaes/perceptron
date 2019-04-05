import pandas  as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import perceptron as p

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
     header=None)

y = df.iloc[0:100, 4].values
X = df.iloc[0:100, [0,2]].values

y = np.where( y == 'Iris-setosa', -1 , 1)

ppn  = p.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

