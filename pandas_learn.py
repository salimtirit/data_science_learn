from numpy.lib.shape_base import column_stack
import pandas as pd
from sklearn.utils.sparsefuncs import mean_variance_axis

url = 'http://bit.ly/imdbratings'
movies = pd.read_csv(url)

#movies.sort_values('duration',ascending=False)
#
#
#movies.head(30)
#
#columns = ['content_rating','duration']
#
#movies.sort_values(columns)

movies.head()

movies.shape

booleans = []

for l in movies.duration:
    if l >= 200:
        booleans.append(True)
    else:
        booleans.append(False)

booleans[0:5]

len(booleans)

is_long = pd.Series(booleans)

movies['genre']

#couldn't understand this really ??
movies[is_long]

#for loopu kaldırmak için

is_long = movies.duration >= 200 

movies[is_long]

movies[movies.duration >= 200].genre

movies.loc[movies.duration >= 200, 'genre']

movies[(movies.duration >= 200) & (movies.genre == 'Drama')]

movies[movies.duration >= 200][movies.genre == 'Drama']

# OR 
movies[(movies.duration >= 200) | (movies.genre == 'Drama')]

(movies.duration >200) | (movies.genre == 'Drama')

# What if you want genres crime, drama, and action?
# slow method
movies[(movies.genre == 'Crime') | (movies.genre == 'Drama') | (movies.genre == 'Action')]

# fast method
filter_list = ['Crime', 'Drama', 'Action']
movies[movies.genre.isin(filter_list)]