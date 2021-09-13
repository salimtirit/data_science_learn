import pandas as pd

df = pd.read_csv('http://bit.ly/autompg-csv')
df.head()

%matplotlib inline

df.plot.scatter(x='hp', y='mpg')