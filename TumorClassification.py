import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.grid_search import GridSearchCV

cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

X = df
y = cancer['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

'''svmmodel = SVC()
svmmodel.fit(X_train,y_train)

pred = svmmodel.predict(X_test)



print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
'''
#Using grid Search

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_estimator_)

gridpred = grid.predict(X_test)

print(confusion_matrix(y_test,gridpred))
print(classification_report(y_test,gridpred))
