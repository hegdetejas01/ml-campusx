import statsmodels.api as sm
import pandas as pd

dataset_url = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/refs/heads/master/data/Advertising.csv"
data = pd.read_csv(dataset_url, index_col=0)
data.columns = [i.lower() for i in data.columns]

X = data[['tv','radio','newspaper']]
X = sm.add_constant(X)

y = data['sales']

model = sm.OLS(y, X).fit()
print(model.summary())