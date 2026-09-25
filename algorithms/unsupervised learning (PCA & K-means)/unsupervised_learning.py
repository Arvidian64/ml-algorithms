import math
import numpy as np


data = np.loadtxt("wine.txt")

X = data[:, -13:]

mean = np.mean(X, axis=0)
std = np.std(X, axis=0, ddof=1)
X_std = (X - mean) / std

# Kovarians-matris
cov_matrix = np.cov(X_std, rowvar=False)

#Eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

#Sortering
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]

coeff_matrix = eigenvectors[:, sorted_indices]

transformed_features = np.dot(X_std, coeff_matrix)

np.savetxt("pca_coefficients.csv", coeff_matrix, delimiter=",", fmt="%.6f")
np.savetxt("wine_pca_transformed.csv", transformed_features, delimiter=",", fmt="%.6f")

explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)
print("explained variance ratio per component:")
for i, ratio in enumerate(explained_variance_ratio, 1):
    print(f"{i}: {ratio:.4f} (cumulative: {np.sum(explained_variance_ratio[:i]):.4f})")



