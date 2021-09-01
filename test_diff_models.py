from math import log
from numpy import power
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn import svm

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

train = pd.read_csv("diabetes.csv", header=0, names=col_names)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = train[feature_cols] # Features
y = train.label # Target variable

X_mid, X_val, y_mid, y_val = train_test_split(X, y, test_size=0.1,random_state=32)
X_train, X_test, y_train, y_test = train_test_split(X_mid, y_mid, test_size=0.22,random_state=32)

res_in_90 = {}


def fun(method):
    method.fit(X_train,y_train)

    y_predicted_one = method.predict(X_test)

    return roc_auc_score(y_test,y_predicted_one)


def fun2(method):
    method.fit(X_mid,y_mid)

    y_predicted_one = method.predict(X_val)

    return roc_auc_score(y_val,y_predicted_one)


svc = svm.SVC()
logreg = LogisticRegression()

dectreeEntropy2 = DecisionTreeClassifier(criterion="entropy",max_depth=2)
dectreeEntropy3 = DecisionTreeClassifier(criterion="entropy",max_depth=3)
dectreeEntropy4 = DecisionTreeClassifier(criterion="entropy",max_depth=4)

dectreeGini2 = DecisionTreeClassifier(max_depth=2)
dectreeGini3 = DecisionTreeClassifier(max_depth=3)
dectreeGini4 = DecisionTreeClassifier(max_depth=4)

randomForest2 = RandomForestClassifier(criterion="entropy",n_estimators=200,n_jobs=-1,max_depth=2)
randomForest3 = RandomForestClassifier(criterion="entropy",n_estimators=200,n_jobs=-1,max_depth=3)
randomForest4 = RandomForestClassifier(criterion="entropy",n_estimators=200,n_jobs=-1,max_depth=4)



first_part = {}

first_part['LogReg']= fun(logreg)
first_part['SVC']=fun(svc)

first_part['DecTreeEntropy 2']=fun(dectreeEntropy2)
first_part['DecTreeEntropy 3']=fun(dectreeEntropy3)
first_part['DecTreeEntropy 4']=fun(dectreeEntropy4)

first_part['DecTreeGini 2']=fun(dectreeGini2)
first_part['DecTreeGini 3']=fun(dectreeGini3)
first_part['DecTreeGini 4']=fun(dectreeGini4)

first_part['Random Forest 2']=fun(randomForest2)
first_part['Random Forest 3']=fun(randomForest3)
first_part['Random Forest 4']=fun(randomForest4)



res_in_90['first_part'] = first_part

second_part = {}

second_part["LogReg"] = cross_val_score(logreg,X_mid,y_mid).mean()
second_part["SVC"] = cross_val_score(logreg,X_mid,y_mid).mean()

second_part["DecTreeEntropy 2"] = cross_val_score(dectreeEntropy2,X_mid,y_mid).mean()
second_part["DecTreeEntropy 3"] = cross_val_score(dectreeEntropy3,X_mid,y_mid).mean()
second_part["DecTreeEntropy 4"] = cross_val_score(dectreeEntropy4,X_mid,y_mid).mean()

second_part["DecTreeGini 2"] = cross_val_score(dectreeGini2,X_mid,y_mid).mean()
second_part["DecTreeGini 3"] = cross_val_score(dectreeGini3,X_mid,y_mid).mean()
second_part["DecTreeGini 4"] = cross_val_score(dectreeGini4,X_mid,y_mid).mean()

second_part["Random Forest 2"] = cross_val_score(randomForest2,X_mid,y_mid).mean()
second_part["Random Forest 3"] = cross_val_score(randomForest3,X_mid,y_mid).mean()
second_part["Random Forest 4"] = cross_val_score(randomForest4,X_mid,y_mid).mean()

res_in_90['second_part'] = second_part



final_part = {}

final_part["LogReg"] = fun2(logreg)
final_part["SVC"] = fun2(svc)
final_part["Random Forest 4"] = fun2(randomForest4)

res_in_90['final_part'] = final_part

param_grid_logreg = {
    'penalty': ['l1','l2', 'elasticnet'],
    'tol': [0.001,0.0001,0.00001],
    'C': [1.0,0.5],
    'fit_intercept': [True,False],
    'max_iter': range(100,700,100)
}

logreg_grid_search = GridSearchCV(logreg,param_grid_logreg,cv=5,scoring="roc_auc",return_train_score=True,verbose=True,n_jobs=-1)