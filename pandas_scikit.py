import pandas as pd

url = 'http://bit.ly/kaggletrain'

train = pd.read_csv(url)


train.head()

feature_cals = ['Pclass', 'Parch']

X = train.loc[:,feature_cals]

X.shape

y = train.Survived

y.shape

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X,y)

url_test = 'http://bit.ly/kaggletest'

test = pd.read_csv(url_test)


# missing Survived column because we are predicting
test.head

X_new = test.loc[:,feature_cals]

X_new.shape

new_pred_class = logreg.predict(X_new)

print(new_pred_class)

# kaggle wants 2 columns
# new_pred_class
# PassengerId

# pandas would align them next to each other
# to ensure the first column is PassengerId, use .set_index
kaggle_data = pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId')
kaggle_data.to_csv('sub.csv')

train.to_pickle('train.pkl')

pd.read_pickle('train.pkl')

pd.read_csv('sub.csv')

ufo_cols = ['city', 'colors reported', 'shape reported', 'state', 'time']
url = 'http://bit.ly/uforeports'
ufo = pd.read_csv(url, names=ufo_cols, header=0)
ufo.columns = ufo.columns.str.replace(' ', '_')
