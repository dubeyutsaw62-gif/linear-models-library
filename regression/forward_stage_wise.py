import pandas as pd
import numpy as np
with open('dataconfig.txt', 'r') as f:
    data_path = f.read().strip()

df = pd.read_csv(data_path)

numerical_df = df.select_dtypes(include=[np.number])
raw_X = numerical_df.iloc[:, :-1].values
y = numerical_df.iloc[:, -1].values  

resudal = y

def corr(X_data, target):
    corr_matrix = np.array([np.corrcoef(X_data[:, i], target)[0, 1] for i in range(X_data.shape[1])])
    corr_matrix = np.nan_to_num(corr_matrix) # Safely handle any zero-variance columns
    c = max(corr_matrix)
    f = np.where(corr_matrix == c)[0]
    f1 = f[0]
    return c, f1

# Get initial best feature
best_value, best_feature = corr(raw_X, y)

for i in range(3):
    N = df.shape[0]
    
    # 1. Grab the chosen column before dropping it
    chosen_column = raw_X[:, best_feature].reshape(N, -1)
    
    # 2. Build your X matrix using the isolated column
    if i==0:
        X = np.hstack((np.ones((N, 1)), chosen_column))
    else :
        X = np.hstack(([chosen_column]))
    X_transpose = X.T
    
    beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ resudal
    if i==0:
        beta_0 = beta[0]
        beta_1 = beta[1]
    else :
        beta_0 = beta[0]
 
    y_pred = X @ beta
    #if np.sum(resudal)>(np.sum(resudal)-np.sum(y_pred)):
    resudal = (resudal - y_pred)
    #else:
        #print(f"break at {i+1}th iteration")
        #break
    
    # 3. FIX: Drop the chosen column from raw_X so it's gone for the next iteration
    raw_X = np.delete(raw_X, best_feature, axis=1)
    
    # Break early if we run out of columns to check
    if raw_X.shape[1] == 0:
        print(f"All features have been exhausted at iteration i = {i}.")
        break
        
    # 4. Find the next best feature from the remaining columns
    best_value, best_feature = corr(raw_X, resudal)

print(resudal)
final_rss = np.sum(resudal ** 2)
print(f"\nFinal Remaining Residual Sum of Squares (RSS): {final_rss:.4f}")