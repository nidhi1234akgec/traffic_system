import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace with your actual dataset)
df = pd.read_csv('trafficdata.csv')

# Display the first few rows of the dataframe
print(df.head())

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])



# Extract useful features from timestamp
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday = 0, Sunday = 6

# Handle categorical columns (if any)
df = pd.get_dummies(df, columns=['weather_condition'], drop_first=True)

# Handle missing values (if any)
df = df.dropna()  # Or use df.fillna() if appropriate

# Display the processed data
print(df.head())

# Define features (X) and target (y)
X = df.drop(columns=['timestamp', 'traffic_volume'])
y = df['traffic_volume']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {X_train.shape}")
print(f"Test data size: {X_test.shape}")

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)






# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Traffic Volume')
plt.ylabel('Predicted Traffic Volume')
plt.title('Random Forest Regression: Actual vs Predicted Traffic Volume')
plt.show()

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning (example)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)



# Save the trained model to a file (traffic_model.pkl)
with open('traffic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training complete. The model is saved as 'traffic_model.pkl'.")
