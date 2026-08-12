import pandas as pd
import numpy as np

with open('dataconfig.txt', 'r') as f:
    data_path = f.read().strip()

df = pd.read_csv(data_path)
DISTINCTION_THRESHOLD = 75
df['Is_Distinction'] = (df['PerformanceIndex'] >= DISTINCTION_THRESHOLD).astype(int)
print(df.shape)

numerical_df = df.select_dtypes(include=[np.number]).drop(columns=['PerformanceIndex'])
raw_X = numerical_df.iloc[:-1, :-1].values
y = numerical_df.iloc[:-1, -1].values  

test_X = numerical_df.iloc[:, :-1].values    
test_y = numerical_df.iloc[:, -1].values

N = len(numerical_df)
X = np.hstack((np.ones((N-1, 1)), raw_X.reshape(N-1, -1), (raw_X**2).reshape(N-1, -1)))  # Add polynomial features if needed
t_x = np.hstack((np.ones((N, 1)), test_X.reshape(N, -1), (test_X**2).reshape(N, -1)))  # Add polynomial features if needed

X_transpose = X.T
beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

raw_predictions = t_x @ beta

predicted_classes = (raw_predictions >= 0.5).astype(int)

misclassifications = np.sum(test_y != predicted_classes)
error_rate = misclassifications / N
accuracy = 1 - error_rate

print("CLASSIFICATION VIA REGRESSION")
print(f"Total rows analyzed: {N}")
print(f"Misclassified points: {misclassifications}")
print(f"Empirical Risk (Error Rate): {error_rate:.2%}")
print(f"Accuracy: {accuracy:.2%}")
