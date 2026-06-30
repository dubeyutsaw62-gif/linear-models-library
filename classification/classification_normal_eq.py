import pandas as pd
import numpy as np

# 1. Load data where the last column is binary labels (0 or 1)
with open('dataconfig.txt', 'r') as f:
    data_path = f.read().strip()

df = pd.read_csv(data_path)
DISTINCTION_THRESHOLD = 75
df['Is_Distinction'] = (df['PerformanceIndex'] >= DISTINCTION_THRESHOLD).astype(int)
print(df.shape)
# Drop the original continuous marks column so the model doesn't cheat!
numerical_df = df.select_dtypes(include=[np.number]).drop(columns=['PerformanceIndex'])
raw_X = numerical_df.iloc[:-1, :-1].values
y = numerical_df.iloc[:-1, -1].values  

test_X = numerical_df.iloc[:, :-1].values    
test_y = numerical_df.iloc[:, -1].values

# 2. Build design matrix (add column of 1s)
N = len(numerical_df)
X = np.hstack((np.ones((N-1, 1)), raw_X.reshape(N-1, -1), (raw_X**2).reshape(N-1, -1)))  # Add polynomial features if needed
t_x = np.hstack((np.ones((N, 1)), test_X.reshape(N, -1), (test_X**2).reshape(N, -1)))  # Add polynomial features if needed

# 3. Solve using your Normal Equation! 
X_transpose = X.T
beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

# Get raw continuous numerical predictions
raw_predictions = t_x @ beta


# This turns the continuous numbers into distinct 0 or 1 class labels
predicted_classes = (raw_predictions >= 0.5).astype(int)

# 6. Compute Error Rate (Zero-One Loss)
misclassifications = np.sum(test_y != predicted_classes)
error_rate = misclassifications / N
accuracy = 1 - error_rate

print("=== NPTEL LECTURE SUMMARY CLASSIFICATION VIA REGRESSION ===")
print(f"Total rows analyzed: {N}")
print(f"Misclassified points: {misclassifications}")
print(f"Empirical Risk (Error Rate): {error_rate:.2%}")
print(f"Accuracy: {accuracy:.2%}")