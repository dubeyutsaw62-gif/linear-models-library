import pandas as pd
import numpy as np

with open('dataconfig.txt', 'r') as f:
    data_path = f.read().strip()

df = pd.read_csv(data_path)

numerical_df = df.select_dtypes(include=[np.number])

# Total features available
all_features = list(range(numerical_df.shape[1] - 1)) 
y = numerical_df.iloc[:, -1].values
N = len(numerical_df)

selected_features = []
current_best_rss = np.inf

print("--- Starting Forward Stepwise Selection ---")

# We want to pick the top k features (let's say we look for up to 4 features max)
max_features_to_select = min(4, len(all_features))

for step in range(max_features_to_select):
    best_new_feature = None
    best_step_rss = np.inf
    best_step_beta = None
    
    # Scan through all features we haven't selected yet
    for feature in all_features:
        if feature in selected_features:
            continue
            
        # Create trial feature subset
        trial_features = selected_features + [feature]
        
        # Pull data columns for this trial subset
        raw_X = numerical_df.iloc[:, trial_features].values
        
        # Build design matrix (Add Column of 1s for the intercept)
        X = np.hstack((np.ones((N, 1)), raw_X.reshape(N, -1)))
        
        # Solve Normal Equation: beta = (X^T * X)^-1 * X^T * y
        try:
            X_transpose = X.T
            beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
            
            # Calculate Residual Sum of Squares (RSS)
            predictions = X @ beta
            rss = np.sum((y - predictions) ** 2)
            
            # Check if this feature is the best performer for this step
            if rss < best_step_rss:
                best_step_rss = rss
                best_new_feature = feature
                best_step_beta = beta
        except np.linalg.LinAlgError:
            # Handle non-invertible matrix cases safely
            continue
            
    # If adding a feature improved our model, lock it in
    if best_new_feature is not None:
        selected_features.append(best_new_feature)
        current_best_rss = best_step_rss
        print(f"Step {step+1}: Added Feature Column Index [{best_new_feature}] -> Remaining RSS: {current_best_rss:.4f}")
    else:
        break

print("\n=== FINAL SELECTION RESULTS ===")
print(f"Greedily Selected Feature Indices: {selected_features}")
print(f"Number of columns chosen: {len(selected_features)} out of {len(all_features)}")