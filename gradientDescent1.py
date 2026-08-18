from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def managePlot(pause=2):
    plt.legend()
    plt.show(block=False)
    plt.pause(pause)
    plt.close()

def calculateYpred(b):
    m=78.35
    return ((m*X.ravel())+b).reshape(4)

def plotGraph(b, pause=2):
    plt.scatter(X,y)
    plt.plot(X, lr.predict(X), color='red', label='OLS_Method')
    plt.plot(X, calculateYpred(b), color='green', label='When b={}'.format(b))
    managePlot(pause)

X,y = make_regression(n_samples=4, n_features=1, n_informative=1, n_targets=1, noise=80, random_state=13)

plt.scatter(X, y)
# managePlot()

lr = LinearRegression()
lr.fit(X,y)
print(lr.coef_)
print(lr.intercept_)

plt.scatter(X,y)
plt.plot(X, lr.predict(X), color='red')
# managePlot()

# now consider slope = 78.35 and the inital guess of b = 0
# apply gradient descent
m = 78.35
b = -100
y_pred = ((m*X)+b).reshape(4)
print(y_pred) # getting the value of y_pred for m=78.35 and b=0

# plotGraph(b)

loss = -2 * np.sum(y - m*X.ravel() - b)
print(loss)

alpha = 0.1
stepSize = alpha*loss
b = b-stepSize
print(b)

# plotGraph(b)

loss = -2 * np.sum(y - m*X.ravel() - b)
print(loss)
stepSize = alpha*loss 
b = b-stepSize
print(b)

# plotGraph(b)

loss = -2 * np.sum(y - m*X.ravel() - b)
print(loss)
stepSize = alpha*loss 
b = b-stepSize
print(b)

# plotGraph(b)



# using loops
m = 78.35
b = -100
epochs = 10
alpha = 0.01
for i in range(epochs):
    loss = -2 * np.sum(y - m*X.ravel() - b)
    stepSize = alpha*loss 
    b = b-stepSize

    plt.plot(X,m*X+b)

plt.scatter(X,y)
managePlot(5)