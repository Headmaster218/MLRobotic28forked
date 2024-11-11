# Import the necessary libraries
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
decision_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
bagging = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=10), n_estimators=50, random_state=42)
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
adaboost = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=10), n_estimators=50, learning_rate=0.1, random_state=42)

# Train the models
decision_tree.fit(X_train, y_train)
bagging.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
adaboost.fit(X_train, y_train)

# Predict using the test set
y_pred_dt = decision_tree.predict(X_test)
y_pred_bagging = bagging.predict(X_test)
y_pred_rf = random_forest.predict(X_test)
y_pred_ada = adaboost.predict(X_test)

# Evaluate the performance of each model
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mse_ada = mean_squared_error(y_test, y_pred_ada)
r2_ada = r2_score(y_test, y_pred_ada)

# Print performance results
print(f"Decision Tree Mean Squared Error: {mse_dt:.4f}, R-squared: {r2_dt:.4f}")
print(f"Bagging Regressor Mean Squared Error: {mse_bagging:.4f}, R-squared: {r2_bagging:.4f}")
print(f"Random Forest Mean Squared Error: {mse_rf:.4f}, R-squared: {r2_rf:.4f}")
print(f"AdaBoost Regressor Mean Squared Error: {mse_ada:.4f}, R-squared: {r2_ada:.4f}")

# Plot the comparison of predicted vs true values for each model
plt.figure(figsize=(12, 8))

# Decision Tree
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_dt, color='blue', label='Decision Tree', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r-', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Decision Tree')
plt.grid(True)

# Bagging Regressor
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_bagging, color='green', label='Bagging', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r-', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Bagging Regressor')
plt.grid(True)

# Random Forest
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_rf, color='orange', label='Random Forest', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r-', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Random Forest')
plt.grid(True)

# AdaBoost
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_ada, color='purple', label='AdaBoost', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r-', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('AdaBoost Regressor')
plt.grid(True)

plt.tight_layout()
plt.show()
