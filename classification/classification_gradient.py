import pandas as pd
import numpy as np

with open('dataconfig.txt', 'r') as f:
    data_path = f.read().strip()

df = pd.read_csv(data_path)

DISTINCTION_THRESHOLD = 75
df['Is_Distinction'] = (df['PerformanceIndex'] >= DISTINCTION_THRESHOLD).astype(int)

numerical_df = df.select_dtypes(include=[np.number]).drop(columns=['PerformanceIndex'])

raw_X = numerical_df.iloc[:-1, :-1].values
y = numerical_df.iloc[:-1, -1].values

test_X = numerical_df.iloc[:, :-1].values
test_y = numerical_df.iloc[:, -1].values

mean = np.mean(raw_X, axis=0)
std = np.std(raw_X, axis=0)

std[std == 0] = 1

raw_X = (raw_X - mean) / std
test_X = (test_X - mean) / std

N_train = len(raw_X)
N_test = len(test_X)

X = np.hstack((np.ones((N_train, 1)), raw_X.reshape(N_train, -1)))
t_x = np.hstack((np.ones((N_test, 1)), test_X.reshape(N_test, -1)))

num_features = X.shape[1]
beta = np.zeros(num_features)

print(beta)

learning_rate = 0.01
iterations = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

for i in range(iterations):
    z = X @ beta
    predictions = sigmoid(z)
    errors = predictions - y
    gradient = (X.T @ errors) / N_train
    beta = beta - (learning_rate * gradient)

print("=== MANUAL LOGISTIC REGRESSION RESULTS ===")
print(f"Optimized Intercept (beta_0): {beta[0]:.4f}")
print(f"Optimized Feature Weights (beta_1 to n): {beta[1:]}\n")

test_probabilities = sigmoid(t_x @ beta)

predicted_classes = (test_probabilities >= 0.5).astype(int)

misclassified = (test_y != predicted_classes).astype(int)
empirical_risk = np.mean(misclassified)

print("=== EVALUATION ON TEST DATA ===")
print(f"Total Test Cases: {N_test}")
print(f"Number of Incorrect Classifications: {np.sum(misclassified)}")
print(f"Empirical Risk (Misclassification Error Rate): {empirical_risk:.2%}")
print(f"Model Accuracy: {(1 - empirical_risk):.2%}")