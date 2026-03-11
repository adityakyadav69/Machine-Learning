# Import library
from sklearn.linear_model import LogisticRegression

# Dataset (Hours studied vs Pass/Fail)
X = [[1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1]

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Prediction
prediction = model.predict([[3.5]])

print("Prediction:", prediction)
