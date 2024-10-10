import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Initialize the Random Forest Regressor
# You can adjust the number of trees (n_estimators) and other parameters for tuning
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# Train the Random Forest Regressor
rf_regressor.fit(X_train, y_train)

# Predict using the test set
y_pred_rf = rf_regressor.predict(X_test)

# Evaluate the model performance
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the performance metrics
print(f"Random Forest Regressor Mean Squared Error: {mse_rf:.4f}")
print(f"Random Forest Regressor R-squared: {r2_rf:.4f}")

# Plot the predictions versus true values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, c='orange', label='Random Forest Predictions', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r-', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Random Forest Regressor: Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()
