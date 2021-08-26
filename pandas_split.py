from inspect import TPFLAGS_IS_ABSTRACT
import numpy as np
import pandas as pd
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#url = 'http://bit.ly/kaggletrain'

#train = pd.read_csv(url)

#X, y = train.loc[:,['Pclass','Parch']], train.Survived
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

train = pd.read_csv("diabetes.csv", header=0, names=col_names)

train.head()

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = train[feature_cols] # Features
y = train.label # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=32)

def fun(param):
    param.fit(X_train,y_train)

    y_predicted = param.predict(X_test)

    accuracyAuto =  roc_auc_score(y_test,y_predicted)

    return accuracyAuto

#####################################################

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)

logistic_regression_acc = fun(logreg)

#####################################################

from sklearn.tree import DecisionTreeClassifier

dcstree = DecisionTreeClassifier(criterion="entropy",max_depth=3)

decision_tree_acc = fun(dcstree)

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus    

dot_data = StringIO()
export_graphviz(dcstree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

#####################################################

from sklearn.tree import ExtraTreeClassifier

extratree = ExtraTreeClassifier()

extra_tree_acc = fun(extratree)


#####################################################

from sklearn.naive_bayes import BernoulliNB

bernoulliNB = BernoulliNB()

bernoulli = fun(bernoulliNB)

#data = pd.DataFrame({'Survived':predicted})
#
#compared =  data.reset_index(drop=True).compare(y_test.to_frame().reset_index(drop=True))
#
#P = y_test.to_frame().value_counts()[1]
#N = y_test.to_frame().value_counts()[0]
#
#FP = compared.Survived.self.value_counts()[1]
#FN = compared.Survived.self.value_counts()[0]
#
#TP = P - FN
#TN = N - FP
#
#accuracyMe = (TP + TN)/(P+N)
#accuracyAuto = accuracy_score(data.reset_index(drop=True),y_test.to_frame().reset_index(drop=True))


