import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split

cancer = datasets.load_breast_cancer()

print("Labels:",cancer.target_names)

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.4, random_state=0)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Roc auc score: ",metrics.roc_auc_score(y_test,y_pred))






