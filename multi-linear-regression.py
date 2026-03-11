# Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset (Temp, Humidity -> Rainfall)
data = {
    "Temperature": [30, 32, 35, 28, 25],
    "Humidity": [70, 65, 80, 75, 85],
    "Rainfall": [200, 180, 250, 220, 300]
}

df = pd.DataFrame(data)

# Independent variables
X = df[["Temperature", "Humidity"]]

# Dependent variable
y = df["Rainfall"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
prediction = model.predict([[29, 78]])

print("Prediction:", prediction)

# Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
