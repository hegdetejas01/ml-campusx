from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r, root_mean_squared_error as rmse
import numpy as np
import random

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)

sgd = SGDRegressor(learning_rate='constant', eta0=0.2)
batchSize = 35

for i in range(100):
    idx = random.sample(range(X_train.shape[0]), batchSize)
    sgd.partial_fit(X_train[idx], y_train[idx])

print("[w1,w2,w3,...,w10]->",sgd.coef_)
print("[w0]->",sgd.intercept_)

y_pred = sgd.predict(X_test)
print("RMSE (same unit) = ", rmse(y_test, y_pred))
print("R2Score = ", r(y_test, y_pred))