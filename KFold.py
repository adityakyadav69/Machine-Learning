from sklearn.model_selection import KFold
import numpy as np

# Dataset
X = np.array([1,2,3,4,5,6,7,8,9,10])

# KFold object
kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X):
    print("Train:", train_index, "Test:", test_index)
