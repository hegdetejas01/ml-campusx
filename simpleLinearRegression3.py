import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, root_mean_squared_error as rmse, r2_score as r


df = pd.read_csv('datasets/placement.csv')
print(df.head(3))

X = df.iloc[:,0:1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
lr = LinearRegression()
lr.fit(X_train, y_train)

plt.scatter(X,y)
plt.plot(X_train, lr.predict(X_train), color='green')
plt.xlabel("CGPA")
plt.ylabel("Package")
plt.title("CGPA versus Package")
plt.show(block=False)
plt.pause(3)
plt.close()


yhat = lr.predict(X_test)
y = y_test.values

print("Mean Absolute Error: ", mae(y, yhat), "LPA")
print("Mean Squared Error: ", mse(y, yhat), "(LPA ^ 2)")
print("Root Mean Squared Error: ",rmse(y, yhat), "LPA")
print("R2Score: ", r(y,yhat))

r2 = r(y,yhat)

# Adjusted R2Score
# A_R2Score = 1 - [[(1-R2Score)(n-1)]/(n-1-k)]
a_r2 = 1 - (((1-r2) * (X_test.shape[0]-1)) / (X_test.shape[0]-1-1))
print("Adjusted R2 Score: ", a_r2)