from itertools import Predicate
import numpy as np
from numpy.core.numeric import cross


from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)

clf = svm.SVC(kernel='linear', C=1, random_state=0)

scores = cross_val_score(clf, X, y, cv=5)###?????

n_samples = X.shape[0]

cv = ShuffleSplit(n_splits=5,test_size=0.1,random_state=0)
cross_val_score(clf, X, y, cv=cv)

predicted = cross_val_predict(clf,X,y, cv=cv )

predicted.shape
