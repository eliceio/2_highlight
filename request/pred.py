import numpy as np
import pickle
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
filename = './svm.sav'
loaded_model = pickle.load(open(filename, 'rb'))
X =  np.load('feat2.npy')
# Simple SVM
y_pred = loaded_model.predict(X)
print(y_pred)
y = np.load('label.npy')
print(y)
