import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

style.use('ggplot')

#reading from excel converting into data frame
df=pd.read_excel("stock_data.xlsx")
df=df.set_index('Date')

#doing basic operation to get "high- low" percentage change
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
#df.set_index('Date', inplace=True)
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#defining the label 
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

#preprocessing of data before applying the algorithm
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])
#defining the trainin set and testing set from data.
# 80% is the traning set and 20% is the testing you can also modify this as per your requirement
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#so we are using linearRegression model
#using all the thread available for processing
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

#this is the score for your algorithm
#you should always go with algorith with the highest score.
confidence = clf.score(X_test, y_test)
print(confidence)

#now using the algorith to predict values
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

#86400 is the number of seconds in one year
#df.set_index('Date', inplace=True)
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

#ploting the prediction on a graph
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

