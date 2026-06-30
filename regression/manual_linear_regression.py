import pandas as pd
import numpy as np

with open('dataconfig.txt', 'r') as f:
    data_path = f.read().strip()

df = pd.read_csv(data_path)
numerical_df = df.select_dtypes(include=[np.number])
test_X = numerical_df.iloc[:,:-1].values  
test_y = numerical_df.iloc[:,-1].values  

raw_X = numerical_df.iloc[:,:-1].values  
y = numerical_df.iloc[:,-1].values  

N = len(numerical_df)
X=np.hstack((np.ones((N, 1)) , raw_X.reshape(N, -1)))
t_x=np.hstack((np.ones((N, 1)) , test_X.reshape(N, -1)))

X_transpose = X.T

# np.linalg.inv computes the matrix inverse
# np.dot or @ performs matrix multiplication
beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

beta_0 = beta[0] # Intercept
beta_1 = beta[1] # Slope for Experience
# beta_2 = beta[2] # Coefficient for Experience squared
# beta_3 = beta[3] # Coefficient for Experience cubed
print("=== MANUAL NORMAL EQUATION RESULTS ===")
print(f"Calculated Intercept (beta_0): {beta_0:.2f}")
print(f"Calculated Slope (beta_1): {beta_1:.2f}")
print(f"Manual Decision Rule: f(x) = {beta_0:.2f} + {beta_1:.2f} * x")
print(f"Calculated Coefficients (beta): {beta}")

# f(X) = X * beta
y_pred = X @ beta

# Calculate Squared Losses manually
squared_errors = (y - y_pred) ** 2

# Mean Squared Error (Empirical Risk)
manual_mse = np.sum(squared_errors)
manual_rmse = np.sqrt(manual_mse)

print(f"rss: {manual_mse:.2f}")
print(f"Manual Interpretability Check (RMSE): {manual_rmse:.2f}")