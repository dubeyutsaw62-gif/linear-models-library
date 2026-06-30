# Linear Models Library: Regression & Classification Fundamentals

Welcome to the **Linear Models Library**. This repository serves as an educational and functional workspace tracking the development of core machine learning algorithms. It covers both **regression** and **classification** workflows, contrasting optimized framework-driven code against deep mathematical, from-scratch manual implementations.

---

## 📂 Repository Structure

```text
├── classification/
│   ├── classification_gradient.py
│   └── classification_normal_eq.py
├── regression/
│   ├── Linear_regression.py
│   ├── manual_linear_regression.py
│   ├── forward_stepwise.py
│   └── forward_stage_wise.py
├── core_apps/
│   ├── app.py
│   └── L_R_web_interface.py
├── .gitignore
└── data_config.txt (Local only)

```

---

## 🧠 Core Mathematical Concepts & Architecture

### 1. Linearity vs. Polynomial Expansion (`np.hstack`)

A powerful realization embedded in this library is that **linear models are not restricted to straight lines**. A model is defined as "linear" because it is linear with respect to its *parameters* (weights), not the input features.

By taking a standard 1D feature matrix $X$ and horizontally stacking higher-degree terms using NumPy:

```python
# Converting Linear Regression into Quadratic/Polynomial Regression
X_quadratic = np.hstack((X, X**2)) 

```

The underlying algorithms in `regression/` automatically transform into **Polynomial Regression** models without modifying a single line of the optimization math.

### 2. Analytical vs. Iterative Solvers

The models demonstrate the two fundamental ways to train a linear system:

* **The Normal Equation (Analytical):** Solving for weights directly in a single mathematical step via $W = (X^T X)^{-1} X^T y$. This provides the exact global minimum instantly but scales poorly with massive feature sets due to matrix inversion overhead ($O(n^3)$).
* **Gradient Descent (Iterative):** Step-by-step optimization using calculated partial derivatives to slowly minimize the loss function. This scales incredibly well to large datasets.

---

## 📄 File-by-File Breakdown

### 🔹 Regression Module (`/regression`)

* **`manual_linear_regression.py`**
* *Concept:* Pure NumPy implementation of Ordinary Least Squares (OLS) and batch gradient descent. Designed to show how residual errors update the weight matrix without relying on framework abstracts.
* *Usage:* Ideal for understanding structural learning rates, cost function monitoring, and initial weight bias configuration.


* **`Linear_regression.py`**
* *Concept:* Leverages production-grade scikit-learn libraries to execute linear mapping. Used as a baseline benchmark to test the speed and accuracy of manual implementations.


* **`forward_stepwise.py`**
* *Concept:* A greedy feature selection technique. It starts with an empty model and iteratively adds the most statistically significant variable one-by-one based on criteria like RSS or Adjusted $R^2$.


* **`forward_stage_wise.py`**
* *Concept:* A more conservative, incremental feature selector. Instead of completely optimizing a coefficient when a feature is added, it changes the coefficient by a tiny step in the correct direction, mimicking a localized Lasso regularization effect.



### 🔹 Classification Module (`/classification`)

* **`classification_normal_eq.py`**
* *Concept:* Adapts linear boundaries for classification tasks using closed-form matrix algebra. Maps raw continuous target thresholds to discrete decision boundary planes.


* **`classification_gradient.py`**
* *Concept:* Implements binary classification optimization iteratively. Features log-loss evaluation and partial derivative steps to separate clustered categorical classifications smoothly.



### 🔹 Core Applications & Deployment (`/core_apps`)

* **`L_R_web_interface.py` & `app.py**`
* *The Pickle-Free Architecture:* Unlike standard deployment scripts that load pre-trained binary models using `pickle` or `joblib`, this UI computes or feeds raw calculated algorithmic weights directly through execution logic.
* *Why this matters:* It eliminates the security vulnerabilities of untrusted `.pkl` execution, bypasses environment serialization mismatches, and keeps the production code entirely transparent and lightweight.



---

## 🔒 Security & Local Data Management

To keep this repository clean, high-performing, and lightweight, all data pipelines utilize local abstraction:

1. **Paths:** The web interfaces read datasets via path routing stored in a local `data_config.txt` or `.env` system.
2. **Exclusions:** Massive raw files (like `spam.csv`) and large binary objects (`*.pkl`) are explicitly blocked via `.gitignore` to prevent repository bloat and speed up command-line performance.

---

```

```
