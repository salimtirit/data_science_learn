from numpy.lib.shape_base import _apply_along_axis_dispatcher
import pandas as pd

link = 'http://bit.ly/uforeports'
ufo = pd.read_csv(link)

cols= ['City','State']

ufo = pd.read_csv(link, usecols = cols)

# reference using position (Integer)
cols2 = [0, 4]

ufo = pd.read_csv(link, usecols=cols2)

# if you only want certain number of rows
ufo = pd.read_csv(link, nrows=3)

link = 'http://bit.ly/drinksbycountry'
drinks = pd.read_csv(link)

drinks.dtypes

import numpy as np

drinks.select_dtypes(include=[np.number]).dtypes

drinks

for index, row in drinks.iterrows():
    print(index, row.country, row.beer_servings)


#Drop non-numeric data in a DataFrame

drinks.dtypes

drinks.select_dtypes(include=[np.number]).dtypes

drinks.describe(include='all')

list_include = ['object', 'float64']
drinks.describe(include=list_include)

drinks.drop('continent', axis=1).head()

drinks.drop(3,axis=0,inplace=True)

drinks.head()

drinks.drop([0,1],axis=0).head()

drinks.mean(axis='index')

#using string methods

url = 'http://bit.ly/chiporders'
orders = pd.read_table(url)

orders.item_name.str.upper()

drinks.beer_servings.astype(float)

drinks = pd.read_csv(link, dtype={'beer_servings':float})

drinks.dtypes

orders.dtypes

orders['item_price'] = orders.item_price.str.replace('$','').astype(float)

orders.dtypes

orders.item_price.mean()

orders[orders.item_name.str.upper().str.contains('Chicken',case=False)]

drinks.beer_servings.mean()

drinks.groupby('continent').beer_servings.mean()

drinks[drinks.continent=='Africa'].head()

drinks.groupby('continent').beer_servings.agg(['count','min','max','mean'])

# allow plots to appear in notebook using matplotlib
%matplotlib inline
#may be important take a look at this
data = drinks.groupby('continent').mean()

data

#important
data.plot(kind='bar')

url = 'http://bit.ly/imdbratings'
movies = pd.read_csv(url)

movies.genre.describe()

movies.genre.value_counts(normalize=True)

movies.genre.unique() #nunique() gives the number of unique values