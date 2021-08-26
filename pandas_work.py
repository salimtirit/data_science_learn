import pandas as pd

mydataset = {
    'cars':['BMW','Volvo','Ford'],
    'passing':[3,7,2]
}

myvar = pd.DataFrame(mydataset)

print(myvar)

print(pd.__version__)

a = [1,7,8]

myvar2 = pd.Series(a)

print(myvar2)

print(myvar2[0])

myvar3 = pd.Series(a,index=["a","b","c"])

print(myvar3)

print(myvar3["b"])

calories = {"day":420,"day2":380,"day3":390}

myvar4 = pd.Series(calories)

print(myvar4)

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

myvar5 = pd.DataFrame(data)

print(myvar5)

#load data into a DataFrame object:
df = pd.DataFrame(data)

print(df)

#refer to the row index:
print(df.loc[0])

print(df.loc[[0,1]])

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

print(df)

print(df.loc["day2"])


df = pd.read_csv('data.csv')

print(df.to_string())

data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}

df = pd.DataFrame(data)

print(df)

##################################################

df = pd.read_csv('data.csv')

#if the number of rows is not specified,
#the head() method will return the top 5 rows.
print(df.head(10))

print(df.tail())

print(df.info())

##################################################

df = pd.read_csv('dirtydata.csv')

new_df = df.dropna()

print(new_df.to_string())

##################################################

df.dropna(inplace=True)

print(df.to_string())

##################################################

df = pd.read_csv("dirtydata.csv")

df.fillna(130,inplace=True)

print(df.to_string())

##################################################

df = pd.read_csv('dirtydata.csv')

df["Calories"].fillna(130, inplace = True)

print(df.to_string())

##################################################

df = pd.read_csv('dirtydata.csv')

x = df["Calories"].mean() # the average value 
#(the sum of all values divided by number of values).

df["Calories"].fillna(x, inplace = True)

##################################################

df = pd.read_csv('dirtydata.csv')

x = df["Calories"].median() # the value in the middle,
# after you have sorted all values ascending.

df["Calories"].fillna(x, inplace = True)

#################################################

df = pd.read_csv('dirtydata.csv')

x = df["Calories"].mode()[0]

df["Calories"].fillna(x, inplace = True)

##################################################

df = pd.read_csv('dirtydata.csv')

#Set "Duration" = 45 in row 7:
df.loc[7, 'Duration'] = 45

##################################################

for x in df.index:
    if df.loc[x, "Duration"] > 120:
        df.loc[x, "Duration"] = 120

##################################################

for x in df.index:
    if df.loc[x, "Duration"] > 120:
        df.drop(x, inplace = True)

##################################################

print(df.duplicated())

df.drop_duplicates(inplace=True)

##################################################

df = pd.read_csv("data.csv")

df.corr()

##################################################


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df.plot()

plt.show()

df.plot(kind = 'scatter', x = 'Duration', y = 'Calories')

plt.show()

df.plot(kind = 'scatter', x = 'Duration', y = 'Maxpulse')

plt.show()

df["Duration"].plot(kind = 'hist')

# reading a well-formatted .tsv file
url = 'http://bit.ly/chiporders'
orders = pd.read_table(url)
orders.head()

url2 = 'http://bit.ly/movieusers'
users = pd.read_table(url2)
users.head()

user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table(url2, sep='|', header=None, names=user_cols)
users.head()