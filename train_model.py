import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
data = pd.read_csv('trafficdata.csv')

# Preprocessing:
# Convert 'DateTime' to datetime and extract features like hour of the day, day of the week, etc.
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['hour'] = data['DateTime'].dt.hour
data['day_of_week'] = data['DateTime'].dt.dayofweek  # 0: Monday, 6: Sunday
data['month'] = data['DateTime'].dt.month  # To capture seasonal trends

# We don't need the 'ID' column for training, so we can drop it.
data.drop('ID', axis=1, inplace=True)

# Encoding 'Junction' column (since it is categorical, we can use Label Encoding or One-Hot Encoding)
# Here, I'll use Label Encoding for simplicity.
label_encoder = LabelEncoder()
data['Junction'] = label_encoder.fit_transform(data['Junction'])

# Features (independent variables) and target (dependent variable)
features = ['hour', 'day_of_week', 'month', 'Junction']
target = 'Vehicles'

X = data[features]  # Independent variables
y = data[target]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model to a file (traffic_model.pkl)
with open('traffic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training complete. The model is saved as 'traffic_model.pkl'.")
