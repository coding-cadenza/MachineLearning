import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris  = load_iris()
data = iris.data
data = pd.DataFrame(data,columns=('x1','x2','x3','y'))
data.to_csv('knn_data_3',index=False)
# data = pd.DataFrame(iris)
