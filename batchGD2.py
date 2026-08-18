# Expected Output from our own GD class

    # [w1,w2,w3,...,w10]-> [-9.15865318, -205.45432163, 516.69374454, 340.61999905, 895.5520019, 561.22067904, 153.89310954, 126.73139688, 861.12700152, 52.42112238]
    # [w0]-> 151.88331005254167
    # RMSE (same unit) =  55.62784079546915
    # R2Score =  0.439933866156897


import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r, root_mean_squared_error as rmse

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)

class GDBatchRegressor:

    def __init__(self, alpha=0.01, epochs=100):
        self.coef = None
        self.intercept = None
        self.alpha = alpha
        self.epochs = epochs

    def fit(self, X, y): 
        # initialise the co-efficient
        self.intercept = 0
        self.coef = np.ones(X.shape[1])

        for i in range(self.epochs):
            # Updating the intercept
            # dL/dw0 = (-2/n) sum(i=1 to n) (yi - yihat)
            yhat = np.dot(X, self.coef) + self.intercept
            derivative_intercept = -2 * np.mean(y - yhat)
            self.intercept = self.intercept - self.alpha*derivative_intercept

            # Updating the co-efficient
            # dL/dwm = -2/n sum(i=0 to n) [(yi - yihat) * Xim]
            # dL/dw (for all w) = -2/n ((yi - yhay) * X)
            derivative_coef = -2 * np.dot((y-yhat), X) / X.shape[0]
            self.coef = self.coef - self.alpha * derivative_coef

        # Printing final answers
        print("[w1,w2,w3,...,w10]->",self.coef)
        print("[w0]->",self.intercept)

    def predict(self, X): 
        return np.dot(X, self.coef) + self.intercept


alpha = 0.1
epoch = 10000
mlr = GDBatchRegressor(alpha, epoch)
mlr.fit(X_train, y_train)

y_pred = mlr.predict(X_test)
print("R2Score = ", r(y_test, y_pred))
print("RMSE = ", rmse(y_test, y_pred))