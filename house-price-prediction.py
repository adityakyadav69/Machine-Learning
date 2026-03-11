# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Size": [1000, 1500, 2000, 2500, 3000],
    "Bedrooms": [2, 3, 3, 4, 4],
    "Price": [200000, 300000, 350000, 400000, 500000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Input features
X = df[["Size", "Bedrooms"]]

# Output
y = df["Price"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price
prediction = model.predict([[1800, 3]])

print("Predicted House Price:", prediction)
