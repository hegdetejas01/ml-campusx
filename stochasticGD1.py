### Code of Batch GD ###
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r, root_mean_squared_error as rmse
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(X_train, y_train)
print("[w1,w2,w3,...,w10]->",lr.coef_)
print("[w0]->",lr.intercept_)

y_pred = lr.predict(X_test)
print("RMSE (same unit) = ", rmse(y_test, y_pred))
print("R2Score = ", r(y_test, y_pred))
print("\n\n SGD \n")



### Code of SGC ###
class SGDRegressor:
    def __init__(self, alpha=0.01, epochs=100):
        self.coef = None
        self.intercept = None
        self.alpha = alpha
        self.epochs = epochs

    def fit(self, X, y):
        # initialise the parameter
        self.intercept = 0
        self.coef = np.ones(X.shape[1])

        for i in range(self.epochs):
            for j in range(X.shape[0]): # code runs for number of example times
                idx = np.random.randint(0, X.shape[0]) # generates random number

                # for the idx-th example[row], we train the model
                yhat = np.dot(X[idx], self.coef) + self.intercept # this is a scalar

                # updating intercept based on a single example
                # dL/dw0 = -2 * (y[idx]-yhat[idx])
                derivative_intercept = -2 * (y[idx] - yhat)
                self.intercept = self.intercept - derivative_intercept*self.alpha

                # updating the co-efficient
                # dL/dwm = -2 * (y[id]-yhat[id]) X[idx](m)
                # -2 ((yi - yhay) * X)
                derivative_coef = -2 * np.dot((y[idx] - yhat), X[idx])
                self.coef = self.coef - derivative_coef*self.alpha

        # Printing final answers
        print("[w1,w2,w3,...,w10]->",self.coef)
        print("[w0]->",self.intercept)

    def predict(self, X): 
        return np.dot(X, self.coef) + self.intercept

alpha = 0.06
epoch = 25
mlr = SGDRegressor(alpha, epoch)
mlr.fit(X_train, y_train)

y_pred = mlr.predict(X_test)
print("RMSE = ", rmse(y_test, y_pred))
print("R2Score = ", r(y_test, y_pred))