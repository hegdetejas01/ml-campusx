# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html

from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r, root_mean_squared_error as rmse

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)

sgd = SGDRegressor(max_iter=100, learning_rate='constant', eta0 = 0.01)
sgd.fit(X_train, y_train)

print(sgd.coef_)
print(sgd.intercept_)

y_pred = sgd.predict(X_test)
print(r(y_test, y_pred))
print(rmse(y_test, y_pred))