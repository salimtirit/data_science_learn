import numpy  as np

from sklearn import random_projection,datasets

from sklearn.svm import SVC
rng = np.random.RandomState(0) #i dont really understand this
X = rng.rand(10,2000)
X = np.array(X,dtype='float32')

X.dtype #when you run this line it says dtype(float32)

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
X_new.dtype

iris = datasets.load_iris()

clf = SVC()
clf.fit(iris.data,iris.target)

list(clf.predict(iris.data[:3]))

clf.fit(iris.data,iris.target_names[iris.target])

list(clf.predict(iris.data[:3]))

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = SVC()
clf.set_params(kernel='linear').fit(X, y)

clf.predict(X[:5])

clf.set_params(kernel='rbf').fit(X, y)

clf.predict(X[:5])