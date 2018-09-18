# coding= UTF-8

import pickle
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd

# Load data from numpy file
X =  np.load('feat.npy')
y =  np.load('label.npy').ravel()

def convert(x):
    if x == 1:
        return('song')
    if x == 2:
        return('speech')

y = map(convert , y)

print('originnal data count :{}'.format(len(X)))

# Split data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print('train data count : {}'.format(len(X_train)))
print('test data count : {}'.format(len(X_test)))


# Simple SVM
print('fitting...')
clf = SVC(C=20.0,probability=True,kernel='rbf',class_weight = {'speech' :1, 'song' : 1}, gamma=0.00001)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = clf.score(X_test, y_test)
print("acc=%0.3f" % acc)
print(pd.crosstab(y_pred, y_test))


filename = 'svm.sav'
pickle.dump(clf, open(filename, 'wb'))

# Grid search for best parameters
# Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
#                      'C': [1, 10 ,20,30,40,50]}]
#                     #  ,
#                     # {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# scores = ['precision', 'recall']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print('')

#     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                        scoring='%s_macro' % score)
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print('')
#     print(clf.best_params_)
#     print('')
#     print("Grid scores on development set:")
#     print('')
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print('')

#     print("Detailed classification report:")
#     print('')
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print('')
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print('')
