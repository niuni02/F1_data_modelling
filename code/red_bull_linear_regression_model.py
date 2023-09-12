# Red Bull Linear Regression Model
# Data source: https://github.com/oracle-devrel/redbull-pit-strategy/tree/main

import os
    # Get the current working directory
cwd = os.getcwd()
    # Print the current working directory
print("Current working directory: {0}".format(cwd))
    # Print the type of the returned object
print("os.getcwd() returns an object of type: {0}".format(type(cwd)))

# Set the desired directory path
path = '/Users/niuni/Desktop/work/GitHub/F1_data_modelling/raw_data'

# Create the directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)

# Change the working directory to the desired one
os.chdir(path)

# Check the working directory is set
cwd = os.getcwd()
    # Print the current working directory
print("Current working directory: {0}".format(cwd))
    # Print the type of the returned object
print("os.getcwd() returns an object of type: {0}".format(type(cwd)))

# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Import csv
data = pd.read_csv("final_data.csv")
data.set_index('Team', inplace=True)
data = data.loc['Red Bull Racing']

# Replace missing values in the 'target' column with 0
data['bestPreRaceTime'].fillna(0, inplace=True)

# Preprocessing and feature engineering
X = data['meanTrackTemp']
X = X.to_numpy().reshape(-1, 1)
y = data['bestPreRaceTime']

# Create polynomial features
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions
y_pred = model.predict(X_poly)

# Calculate the mean squared error
mse = mean_squared_error(y, y_pred)

# Plot the original data and the regression curve
plt.scatter(X, y, label="Original Data")
plt.plot(X, y_pred, color='red', label="Polynomial Regression")
plt.xlabel("Mean TrackTemperature (Celsius)")
plt.ylabel("Best Pre Race Time")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

print("Mean Squared Error:", mse)