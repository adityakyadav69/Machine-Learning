import pandas as pd

# Dataset
data = {
    "City": ["Delhi", "Mumbai", "Delhi", "Chennai"]
}

df = pd.DataFrame(data)

# Convert categorical data to dummy variables
dummy = pd.get_dummies(df["City"])

print(dummy)
