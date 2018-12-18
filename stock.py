import pandas as pd
import quandl as Q
import math
import numpy as np
import sklearn



#Accessing Quandl Database

df=Q.get("SSE/GGQ1", authtoken="")
#print (df.head())

#Storinf it in a Dataframe

df=df[['High','Low','Last']]

#Basic Calculation of Percent Change in the stock price 
df['Percent'] = ((df['High']-df['Low'])/df['Low'])*100

#Storing the Percent Change and the Last seen Price of the Stock
df=df[['Last','Percent']]


forecast_col='Last'
df.fillna('-99999',inplace=True)

forecast_out=int(math.ceil(0.01*len(df)))

df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

##features
X=np.array(df.drop(['label'],1))
#Labels
y=np.array(df['label'])



X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,y,test_size=0.2)

clf=LinearRegression()
clf.fit(X_train,Y_train)
acc=clf.score(X_test,Y_test)

print(acc)

