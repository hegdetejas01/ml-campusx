import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# This is simple linear regression and doesn't handle multile input feature

class MyLinearRegression:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X, y):
    # calculate m and b
        # m = sum((xi-xbar)(yi-ybar))/sum((xi-xbar)^2)
        xbar = X.mean()
        ybar = y.mean()
        numerator = np.sum((X - xbar) * (y - ybar))
        denominator = np.sum((X - xbar)**2)
        self.m = numerator/denominator
        print("m = " + str(self.m))

        # b = ybar - (m*xbar)
        self.b = ybar - (self.m * xbar)
        print("b = " + str(self.b))

        return self.m, self.b
    
    def predict(self, X):
        # y = mx+b
        print("Testing for x =", X)
        return (self.m * X) + self.b


df = pd.read_csv('datasets/placement.csv')
print(df.head(2))

X = df.iloc[:,0].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

mlr = MyLinearRegression()
m,b = mlr.fit(X_train, y_train)

for i in range(X_test.shape[0]):
    y = mlr.predict(X_test[i])
    print("Predicted Value yhat =",y,"\tExpected Value y=",y_test[i],"\tError =",y_test[i] - y,"\n")