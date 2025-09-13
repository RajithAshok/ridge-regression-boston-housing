import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

# Load Boston Housing dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data.values
y = boston.target.values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with feature scaling, polynomial features, and ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge_reg', Ridge())
])

# Set hyperparameter grid for ridge alpha (regularization strength)
param_grid = {'ridge_reg__alpha': [0.1, 1.0, 10.0, 100.0]}

# Use GridSearchCV to find best alpha
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_
print(f"Best Ridge alpha: {grid_search.best_params_['ridge_reg__alpha']}")

# Predict and evaluate on test set
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Improved Model R²: {r2:.3f}")
print(f"Improved Model MSE: {mse:.2f}")

# Plot true vs predicted prices
plt.figure(figsize=(7,4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("Improved Ridge Regression: Boston Housing\nTrue vs Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()
