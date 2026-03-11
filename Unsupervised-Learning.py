from sklearn.cluster import KMeans
import numpy as np

# Data
X = np.array([[1,2], [1,4], [1,0], [10,2], [10,4], [10,0]])

# Model
kmeans = KMeans(n_clusters=2)

# Train model
kmeans.fit(X)

# Predict clusters
print(kmeans.labels_)
