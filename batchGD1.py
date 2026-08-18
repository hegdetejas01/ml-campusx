from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r, root_mean_squared_error as rmse

X, y = load_diabetes(return_X_y=True)

# there are 442 examples, and 10 features
print(X.shape)
print(y.shape)  
# What is there in the data
print(X[:3,])
print(y[:3])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)
lr = LinearRegression()
lr.fit(X_train, y_train)
print("[w1,w2,w3,...,w10]->",lr.coef_)
print("[w0]->",lr.intercept_)

y_pred = lr.predict(X_test)
print("RMSE (same unit) = ", rmse(y_test, y_pred))
print("R2Score = ", r(y_test, y_pred))