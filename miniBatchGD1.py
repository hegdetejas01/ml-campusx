
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r, root_mean_squared_error as rmse
import numpy as np
import random

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)

### Batch GD ###
lr = LinearRegression()
lr.fit(X_train, y_train)
print("[w1,w2,w3,...,w10]->",lr.coef_)
print("[w0]->",lr.intercept_)

y_pred = lr.predict(X_test)
print("RMSE (same unit) = ", rmse(y_test, y_pred))
print("R2Score = ", r(y_test, y_pred))
print("\n\n Mini Batch \n")



### Mini Batch GD ###
class MiniBatchGD:
    def __init__(self, batchSize, alpha, epochs):
        self.batchSize = batchSize
        self.alpha = alpha
        self.epochs = epochs
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        self.intercept = 0
        self.coef = np.ones(X.shape[1])

        for i in range(self.epochs):
            for j in range(int(X.shape[0]/self.batchSize)): # runs the loop as the number of batches
                idx = random.sample(range(X.shape[0]), self.batchSize)
                yhat = np.dot(X[idx], self.coef) + self.intercept

                derivative_intercept = -2 * np.mean(y[idx] - yhat)
                self.intercept -= self.alpha*derivative_intercept

                derivative_slope = -2 * np.dot((y[idx] - yhat), X[idx])
                self.coef -= self.alpha*derivative_slope

        # Printing final answers
        print("[w1,w2,w3,...,w10]->",self.coef)
        print("[w0]->",self.intercept)
        
    def predict(self, X): 
        return np.dot(X, self.coef) + self.intercept

alpha = 0.01
batchSize = int(X_train.shape[0]/10)
epochs = 100
mlr = MiniBatchGD(batchSize=batchSize, alpha=alpha, epochs=epochs)
mlr.fit(X_train, y_train)

print("[w1,w2,w3,...,w10]->",mlr.coef)
print("[w0]->",mlr.intercept)

y_pred = mlr.predict(X_test)
print("RMSE (same unit) = ", rmse(y_test, y_pred))
print("R2Score = ", r(y_test, y_pred))