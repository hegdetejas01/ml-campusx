from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, root_mean_squared_error as rmse, r2_score as r

X, y = load_diabetes(return_X_y=True)
# print(X)
# print(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
print(X_train.shape)
print(X_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Beta = \t\t", lr.coef_)
print("Intercept = \t\t", lr.intercept_)

y_pred = lr.predict(X_test)

# for i in range(X_test.shape[0]):
#     print("For x=", X_test[i])
#     print(y_test[i], "-->", y_pred[i], "\tdiff-->", y_pred[i] - y_test[i])
#     print()

print("MSE:\t",mse(y_test, y_pred))
print("MAE:\t",mae(y_test, y_pred))
print("RMSE:\t",rmse(y_test, y_pred))
print("r2S:\t",r(y_test, y_pred))


# MSE:     3094.4566715660617
# MAE:     45.21303419046902
# RMSE:    55.62784079546915
# r2S:     0.439933866156897

# Beta =           [  -9.15865318 -205.45432163  516.69374454  340.61999905 -895.5520019
#   561.22067904  153.89310954  126.73139688  861.12700152   52.42112238]
# Intercept =              151.88331005254167