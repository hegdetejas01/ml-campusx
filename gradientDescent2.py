from mpl_toolkits.mplot3d import axes3d
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error as rmse

X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=13)
plt.scatter(X,y)
# plt.show()

lr = LinearRegression()
lr.fit(X,y)
print("Slope = ", lr.coef_)
print("Intercept = ", lr.intercept_)

plt.plot(X, lr.predict(X), color='red')
plt.show()

print("RMSE (Same unit by sklearn) = ", rmse(y, y_pred=(lr.predict(X))))

print("Cross Validation Score = ", np.mean(cross_val_score(lr, X, y, scoring='r2', cv=10)))

class GDRegressor:
    def __init__(self, alpha, epochs):
        self.m = 100
        self.b = -120
        self.alpha = alpha
        self.epochs = epochs

    @staticmethod
    def plotGraph(X, Y, xlabel, ylabel, title):
        plt.plot(X,Y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    def fit(self, X, y):

        initialCost = np.sum((y - (self.m * X.ravel()) - self.b)**2)
        iValue = [0]
        bValue = [self.b]
        cost = [initialCost]
        mValue = [self.m]

        # calculate b and m with the help of gradient descent
        for i in range(self.epochs):
            print("Epochs = ", i+1)

            # b_slope = dL/db = -2 * sum(i=1 to n) (yi - mxi - b)
            # m_slope = dL/dm = -2 * sum(i=1 to n) (yi - mxi - b)(xi)
            loss_slope_b = -2 * np.sum(y - self.m*X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((y - self.m*X.ravel() - self.b) * X.ravel())

            # b = b - alpha * slope
            # m = m - alpha * slope
            self.b = self.b - (self.alpha * loss_slope_b)
            self.m = self.m - (self.alpha * loss_slope_m)

            print("\tB = ",self.b, end="\t\t")
            print("\tM = ",self.m, end="\t\t")

            loss = np.sum((y - (self.m * X.ravel()) - self.b)**2)/X.shape[0]
            print("\tLOSS = ", loss, "\n")

            cost.append(loss)
            iValue.append(i+1)
            bValue.append(self.b)
            mValue.append(self.m)

        print("Final value of B = ", self.b)
        print("Final value of M = ", self.m)
        print("Final Loss = ", np.sum((y - (self.m * X.ravel()) - self.b)**2)/X.shape[0])
        print("RMSE (same unit): ", rmse(y, y_pred=(self.m * X.ravel()) + self.b))

        self.plotGraph(iValue, bValue, "Epoch", "B", "Epoch Versus B")
        self.plotGraph(iValue, mValue, "Epoch", "M", "Epoch Versus M")
        self.plotGraph(bValue, cost, "B", "Cost", "B Versus Cost")
        self.plotGraph(mValue, cost, "M", "Cost", "M Versus Cost")
        self.plotGraph(cost, iValue, "Cost", "Epoch", "Cost Versus Epoch")

    def predict(self, X):
        return self.m * X.ravel() + self.b

alpha = 0.003
epochs = 100
gd = GDRegressor(alpha, epochs)
gd.fit(X,y) 

# Slope =  [27.82809103]
# Intercept =  -2.29474455867698