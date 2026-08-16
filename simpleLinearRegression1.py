"""
Original file is located at
    https://colab.research.google.com/drive/1TJp3VbStsMWx7BUAn9Z_MF9ZmT_Bd7np
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('datasets/placement.csv')
print(df.head(3))

X = df.iloc[:,0:1]
y = df.iloc[:,-1]
print(y)


#  y = mx + b
#  package = m * cgpa + b
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)
print(X_test)
print(y_test)

lr = LinearRegression()
lr.fit(X_train, y_train) # trains the model

print(lr.predict(X_test.iloc[0].values.reshape(1,1)))

plt.scatter(df['cgpa'], df['package'])
plt.plot(X_test, lr.predict(X_test), color='red')
plt.xlabel("CGPA")
plt.ylabel("PACKAGE (in LPA)")
plt.title("CGPA versus PACKAGE")
plt.show(block=False)
plt.pause(3)
plt.close()


m = lr.coef_ # value of m
b = lr.intercept_ # value of b
print(b)
print(m)


x = X_test.iloc[0].values
y = m * x + b
print(y)