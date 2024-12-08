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
df['DateTime'] = pd.to_datetime(df['DateTime'])

#  Deleting the ID column
df=df.drop(["ID"], axis=1)



# Extract useful features from timestamp
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek  # Monday = 0, Sunday = 6
df["Date_no"] = df["DateTime"].dt.day


# Pivoting Dateset from junctions
df_junction = df.pivot(columns = "Junction", index="DateTime")


# Creating new datasets 
df_1 = df_junction[[('Vehicles', 1)]]  
df_2 = df_junction[[('Vehicles', 2)]]  
df_3 = df_junction[[('Vehicles', 3)]]  
df_4 = df_junction[[('Vehicles', 4)]]  
df_4 = df_4.dropna() #For only a few months, Junction 4 has only had minimal data.  
  

# As DFS's data frame contains many indices, its index is lowering level one.  
list_dfs = [df_1, df_2, df_3, df_4]  
for i in list_dfs:  
    i.columns= i.columns.droplevel(level=1)  



# Normalize Function  
def Normalize(dataframe,column):  
    average = dataframe[column].mean()  
    stdev = dataframe[column].std()  
    df_normalized = (dataframe[column] - average) / stdev  
    df_normalized = df_normalized.to_frame()  
    return df_normalized, average, stdev  
  
# Differencing Function  
def Difference(dataframe,column, interval):  
    diff = []  
    for i in range(interval, len(dataframe)):  
        value = dataframe[column][i] - dataframe[column][i - interval]  
        diff.append(value)  
    return diff  




# In order to make the series stationary, normalize and differ  
df_N1, avg_J1, std_J1 = Normalize(df_1, "Vehicles")  
Diff_1 = Difference(df_N1, column="Vehicles", interval=(24*7)) #taking a week's difference  
df_N1 = df_N1[24*7:]  
df_N1.columns = ["Norm"]  
df_N1["Diff"]= Diff_1  
  
df_N2, avg_J2, std_J2 = Normalize(df_2, "Vehicles")  
Diff_2 = Difference(df_N2, column="Vehicles", interval=(24)) #taking a day's difference  
df_N2 = df_N2[24:]  
df_N2.columns = ["Norm"]  
df_N2["Diff"]= Diff_2  
  
df_N3, avg_J3, std_J3 = Normalize(df_3, "Vehicles")  
Diff_3 = Difference(df_N3, column="Vehicles", interval=1) #taking an hour's difference  
df_N3 = df_N3[1:]  
df_N3.columns = ["Norm"]  
df_N3["Diff"]= Diff_3  
  
df_N4, avg_J4, std_J4 = Normalize(df_4, "Vehicles")  
Diff_4 = Difference(df_N4, column="Vehicles", interval=1) #taking an hour's difference  
df_N4 = df_N4[1:]  
df_N4.columns = ["Norm"]  
df_N4["Diff"]= Diff_4  










# Handle missing values (if any)
df = df.dropna()  # Or use df.fillna() if appropriate

# Display the processed data
print(df.head())

# Define features (X) and target (y)
X = df.drop(columns=["DateTime", 'Vehicles'])
y = df['Vehicles']

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
