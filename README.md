# Boston Housing Price Prediction with Ridge Regression

This project builds a machine learning model to predict Boston housing prices using Ridge Regression with polynomial features. The model improves upon simple linear regression by capturing nonlinear relationships and applying regularization for better performance.

---

## Project Overview

- Dataset: Boston Housing dataset (506 samples, 13 features)
- Model: Ridge Regression with polynomial features (degree 2) and feature scaling
- Hyperparameter tuning: Grid search with 5-fold cross-validation to select regularization strength (alpha)
- Evaluation metrics: Coefficient of Determination (R²) and Mean Squared Error (MSE)
- Result: Achieved improved R² of 0.818 and reduced MSE of 13.34 on the test set

---

## Training & Evaluation Output

Example terminal output showing best ridge alpha selected and model performance metrics:

<img width="426" height="107" alt="TerminalOutput" src="https://github.com/user-attachments/assets/c7903cc3-3da5-4b57-9fbb-2ec3d006e1c1" />



---

## True vs Predicted Prices Plot

Visualization comparing true house prices against the model's predicted values on the test set:

<img width="600" height="400" alt="TrueVsPredictedPlot" src="https://github.com/user-attachments/assets/b9c072df-648c-4359-9c2f-b31ae0c0aeaa" />


---

## How To Run

1. Clone this repository.
2. Install Python 3.x and required packages:
```
pip install numpy matplotlib scikit-learn
```
3. Run the `boston_housing.ipynb` notebook.

---

Feel free to explore, modify, and improve the model. Pull requests and suggestions are welcome!
