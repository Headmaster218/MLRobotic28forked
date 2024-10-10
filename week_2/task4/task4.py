import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate or reuse synthetic data
# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the AdaBoost Regressor with Decision Tree as the base estimator
# You can adjust the number of estimators (n_estimators) and learning rate for tuning
ada_regressor = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), 
                                  n_estimators=50, learning_rate=0.1, random_state=42)

# Train the AdaBoost Regressor
ada_regressor.fit(X_train, y_train)

y_train_ada = ada_regressor.predict(X_train)
# Predict using the test set
y_pred_ada = ada_regressor.predict(X_test)

# Evaluate the model performance
mse_ada = mean_squared_error(y_train, y_train_ada)
r2_ada = r2_score(y_test, y_pred_ada)

# Print the performance metrics
print(f"AdaBoost Regressor Mean Squared Error: {mse_ada:.4f}")
print(f"AdaBoost Regressor R-squared: {r2_ada:.4f}")

# Plot the predictions versus true values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ada, c='purple', label='AdaBoost Predictions', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r-', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('AdaBoost Regressor: Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()
