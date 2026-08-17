import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, root_mean_squared_error as rmse, r2_score as r

class MyMultipleLR:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1) # adding 1 in the 0th position as per the equation

        # calculating W
        # W = (XT X)(inverse) XT Y
        W = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
        self.coef = W[1:]
        self.intercept = W[0]

    def predict(self, X):
        # Yhat = XW + intercept
        y_pred = np.dot(X, self.coef) + self.intercept
        return y_pred

    def print(self):
        print("Beta = \t\t", self.coef)
        print("Intercept = \t\t", self.intercept)

X, y = load_diabetes(return_X_y=True)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
print(X_train.shape)
print(X_test.shape)

mlr = MyMultipleLR()
mlr.fit(X_train, y_train)
mlr.print()
        
y_pred = mlr.predict(X_test)

print("MSE:\t",mse(y_test, y_pred))
print("MAE:\t",mae(y_test, y_pred))
print("RMSE:\t",rmse(y_test, y_pred))
print("r2S:\t",r(y_test, y_pred))

# MSE:     3094.4566715660662
# MAE:     45.21303419046905
# RMSE:    55.62784079546919
# r2S:     0.4399338661568961

# Beta =           [  -9.15865318 -205.45432163  516.69374454  340.61999905 -895.5520019
#   561.22067904  153.89310954  126.73139688  861.12700152   52.42112238]
# Intercept =              151.8833100525417