import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

with open('dataconfig.txt', 'r') as f:
    data_path = f.read().strip()

df = pd.read_csv(data_path)
# d3=df.to_numpy()
# print(d3)
numerical_df = df.select_dtypes(include=[np.number])
X = numerical_df.iloc[:,:-1].values  # All columns except the last one
y = numerical_df.iloc[:,-1].values  # The last column is the target variable

model = LinearRegression()
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
mse = -scores
expected_risk = np.mean(mse)
print(f"Mean Squared Error (MSE) from Leave-One-Out Cross-Validation: {expected_risk:.2f}")
model.fit(X, y)
intercept = model.intercept_
coefficient = model.coef_[0]
print(f"Intercept: {intercept:.2f}")
print(f"Coefficient: {coefficient:.2f}")